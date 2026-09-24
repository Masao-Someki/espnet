"""Dataset classes for ESPnet3."""

import copy
import logging
from abc import ABC
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from torch.utils.data.dataset import Dataset

from espnet3.components.data.dataset_uid import DatasetUidEntry, format_uid, parse_uid
from espnet3.utils.logging_utils import build_callable_name, build_qualified_name

logger = logging.getLogger(__name__)

_UNRESOLVABLE = object()


def do_nothing(*x):
    """Return input as-is.

    Args:
        x: Any object.

    Returns:
        The input object unchanged.
    """
    if len(x) == 1:
        return x[0]
    else:
        return x


class CombinedDataset:
    """Combines multiple datasets into a single unified dataset-like interface.

    This class supports seamless access to multiple datasets as if they were one.
    Each dataset can be paired with a transform and a global preprocessor, which are
    applied sequentially to each sample. It also supports optional UID handling for
    ESPnet-style preprocessing.

    CombinedDataset supports two indexing modes:
        * Numeric mode (default): every underlying dataset accepts integer indices
          and the combined dataset behaves like a contiguous sequence.
        * String mode: if any dataset requires string-based utterance IDs, the
          organizer builds a lookup table mapping every UID to its source dataset
          while preserving DataLoader-friendly integer access.

    **Dataset-hash UID protocol.** ``get_uid(idx)`` is the single source of
    truth for the utterance ID used from ``collect_stats`` (shape-file keys)
    through batching (the UID returned alongside each sample when
    ``use_espnet_collator`` is set, and passed to the preprocessor). A UID has
    the form ``"<8-hex-char dataset hash>:<position>"``: the hash identifies a
    sub-dataset by its config (via ``espnet3.components.data.dataset_uid``,
    ignoring keys ending in ``_dir``), and ``position`` is that item's index
    within that sub-dataset. Because the hash depends only on the entry's own
    config, the UID is stable across reordering or adding sibling entries --
    it never encodes where a sub-dataset sits among its siblings.
    ``DataOrganizer`` computes ``uid_prefixes``/``uid_entries`` from each
    entry's config and passes them to this constructor; that is the only
    supported way to get stable UIDs. A ``CombinedDataset`` built directly
    without ``uid_prefixes`` (``uid_prefixes=None``, the default) falls back
    to the legacy ``str(global_idx)`` UID, which is **not** stable across
    reordering, for backward compatibility only.

    Args:
        datasets (List[Any]): A list of dataset instances. Each must implement
            `__getitem__` and `__len__`.
        transforms (List[Tuple[Callable, Callable]]): A list of
            (transform, preprocessor) tuples. Each pair corresponds to the matching
            dataset in `datasets`.
            - `transform(sample)` is applied first.
            - Then `preprocessor(uid, sample)` or `preprocessor(sample)` is applied,
              depending on `use_espnet_preprocessor`.
        use_espnet_preprocessor (bool): If True, applies the preprocessor as
            `preprocessor(uid, sample)`. This is used for ESPnet `AbsPreprocessor`
            compatible pipelines.
        uid_prefixes (Optional[List[str]]): One 8-hex-char dataset hash per
            entry in `datasets`, in the same order. Normally supplied by
            `DataOrganizer`. `None` (the default) selects the legacy
            `str(global_idx)` UID scheme for backward compatibility.
        uid_entries (Optional[List[DatasetUidEntry]]): The `DatasetUidEntry`
            per sub-dataset (same order as `datasets`/`uid_prefixes`), used to
            write the `collect_stats` UID table and to build the fingerprint.
            `None` when `uid_prefixes` is `None`.

    Note:
        At initialization, the first sample from each dataset is passed through
        its associated transform to check that all datasets produce dictionaries
        with the same set of keys. This ensures consistency across the combined dataset.
        An `AssertionError` is raised if the keys differ.

    Raises:
        IndexError: If a requested index is outside the range of the combined dataset.
        ValueError: If index is a non-integer string that none of the underlying
            datasets accept as an utterance ID.
        RuntimeError: If `shard()` is called but not supported.
        AssertionError: If output keys from different datasets are inconsistent.

    Example:
        >>> dataset = CombinedDataset(
        ...     datasets=[ds1, ds2],
        ...     transforms=[
        ...         (transform1, preprocessor),
        ...         (transform2, preprocessor),
        ...     ],
        ...     use_espnet_preprocessor=True
        ... )
        >>> sample = dataset[5]
        >>> print(sample["text"])
    """

    def __init__(
        self,
        datasets: List[Any],
        transforms: List[Tuple[Callable, Callable]],
        use_espnet_preprocessor: bool = False,
        uid_prefixes: Optional[List[str]] = None,
        uid_entries: Optional[List[DatasetUidEntry]] = None,
    ):
        """Initialize CombinedDataset object."""
        self.datasets = datasets
        self.transforms = []
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = []
        self.use_espnet_preprocessor = use_espnet_preprocessor

        for transform, preprocessor in transforms:
            if transform is None:
                transform = do_nothing
            if preprocessor is None:
                preprocessor = do_nothing
            assert callable(transform), "transform must be callable."
            assert callable(preprocessor), "preprocessor must be callable."
            self.transforms.append((transform, preprocessor))

        total = 0
        for length in self.lengths:
            total += length
            self.cumulative_lengths.append(total)

        self._string_index_mode = False
        self._string_key_to_dataset: Dict[str, Tuple[int, Any]] = {}
        self._dataset_supports_int: List[bool] = []
        self._dataset_key_lists: List[Optional[List[str]]] = []

        self._initialize_index_mode()

        self._uid_prefixes = uid_prefixes
        self._uid_entries = uid_entries
        self._prefix_to_dataset: Dict[str, int] = (
            {prefix: i for i, prefix in enumerate(uid_prefixes)}
            if uid_prefixes is not None
            else {}
        )
        self._position_maps: List[Optional[Sequence[int]]] = [None] * len(datasets)
        self._shard_index: Optional[int] = None

        # Check the first sample from all dataset to ensure they all have the same keys
        sample_keys = None
        for i, (dataset, transform) in enumerate(zip(self.datasets, self.transforms)):
            if len(dataset) == 0:
                continue  # Skip empty datasets

            reference_key = self._select_reference_key_for_dataset(i)
            sample = transform[0](copy.deepcopy(dataset[reference_key]))
            keys = set(sample.keys())
            if sample_keys is None:
                sample_keys = keys
            else:
                assert keys == sample_keys, (
                    f"Inconsistent output keys in dataset {i}: "
                    f"{keys} != {sample_keys}"
                )

        # Check if dataset is a subclass of ShardedDataset.
        has_sharded = any(
            isinstance(dataset, ShardedDataset) for dataset in self.datasets
        )
        if has_sharded and not all(
            isinstance(dataset, ShardedDataset) for dataset in self.datasets
        ):
            raise RuntimeError(
                "If any dataset is a subclass of ShardedDataset,"
                " then all dataset should be a subclass of ShardedDataset."
            )
        if has_sharded:
            total_shards_set = {
                getattr(dataset, "total_shards", None) for dataset in self.datasets
            }
            dist_world_size_set = {
                getattr(dataset, "dist_world_size", None) for dataset in self.datasets
            }
            if None in total_shards_set or None in dist_world_size_set:
                raise RuntimeError(
                    "ShardedDataset requires total_shards and dist_world_size "
                    "to be set."
                )
            if len(total_shards_set) != 1 or len(dist_world_size_set) != 1:
                raise RuntimeError(
                    "All sharded datasets must share the same total_shards and "
                    "dist_world_size."
                )
            self.total_shards = total_shards_set.pop()
            self.dist_world_size = dist_world_size_set.pop()

        # This flag will be overrode by ESPnetLightningModule.
        self._use_espnet_collator = False

    @property
    def use_espnet_collator(self):
        """Get the flag indicating whether to use ESPnet collator."""
        return self._use_espnet_collator

    @use_espnet_collator.setter
    def use_espnet_collator(self, value: bool):
        """Set the flag indicating whether to use ESPnet collator."""
        self._use_espnet_collator = value

    def __len__(self):
        """Return the total number of samples in the combined dataset."""
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        """Return the item at the given index from the appropriate sub-dataset."""
        if isinstance(idx, str) and self._uid_prefixes is not None:
            parsed = parse_uid(idx)
            if parsed is not None:
                return self._getitem_by_uid(idx, parsed)

        if self._string_index_mode:
            return self._getitem_string_mode(idx)

        if isinstance(idx, str):
            try:
                numerical_idx = int(idx)
            except (ValueError, TypeError):
                return self._getitem_by_utterance_id(idx)
            else:
                idx = numerical_idx

        return self._getitem_int(idx)

    def _getitem_by_uid(self, uid: str, parsed: Tuple[str, int]):
        """Resolve a well-formed dataset-hash UID to its item."""
        prefix, pos = parsed
        dataset_idx = self._prefix_to_dataset.get(prefix)
        if dataset_idx is None:
            raise KeyError(
                f"UID {uid!r}: dataset hash {prefix!r} is not part of this "
                "dataset. The shape files were produced for a different "
                "dataset configuration; re-run collect_stats."
            )
        if self._position_maps[dataset_idx] is not None:
            raise RuntimeError(
                "UID lookup on a sharded dataset is not supported; use "
                "integer indices for shard-local access."
            )
        if not 0 <= pos < self.lengths[dataset_idx]:
            raise IndexError(
                f"UID {uid!r}: position {pos} is out of range for dataset "
                f"{dataset_idx} (length {self.lengths[dataset_idx]})."
            )
        global_idx = (
            self.cumulative_lengths[dataset_idx - 1] if dataset_idx else 0
        ) + pos
        return self._getitem_int(global_idx)

    def _getitem_int(self, idx: int):
        """Return the item at global integer index ``idx``."""
        dataset_idx, ds_idx = self._resolve_global_index(idx)
        try:
            sample = self.datasets[dataset_idx][ds_idx]
        except Exception as e:
            raise RuntimeError(
                f"Failed to access dataset at index {dataset_idx} or "
                f"item at index {ds_idx}. "
                f"Original error: {e}"
            ) from e

        transformed = self.transforms[dataset_idx][0](sample)  # apply transform
        uid = self.get_uid(idx)
        if self.use_espnet_preprocessor:
            transformed = self.transforms[dataset_idx][1](uid, transformed)
        else:
            transformed = self.transforms[dataset_idx][1](transformed)

        if self.use_espnet_collator:
            return uid, transformed
        else:
            return transformed

    def _getitem_by_utterance_id(self, uid: str):
        if self._string_index_mode:
            return self._getitem_string_mode(uid)

        last_error = None
        for dataset, (transform, preprocessor) in zip(self.datasets, self.transforms):
            try:
                sample = dataset[uid]
            except (KeyError, TypeError, ValueError, IndexError) as err:
                last_error = err
                continue

            transformed = transform(sample)
            if self.use_espnet_preprocessor:
                transformed = preprocessor(uid, transformed)
            else:
                transformed = preprocessor(transformed)

            if self.use_espnet_collator:
                return uid, transformed
            return transformed

        raise ValueError(
            f"Utterance ID '{uid}' is not supported by the underlying datasets."
        ) from last_error

    # ------------------------------------------------------------------
    # Internal helpers for string-index mode
    # ------------------------------------------------------------------
    def _initialize_index_mode(self):
        """Determine whether datasets should be accessed via string keys."""

        def supports_integer_index(dataset):
            try:
                dataset[0]
            except Exception:
                return False
            else:
                return True

        self._dataset_supports_int = []
        for dataset in self.datasets:
            self._dataset_supports_int.append(supports_integer_index(dataset))

        all_integer_addressable = all(self._dataset_supports_int)
        self._dataset_key_lists = [None] * len(self.datasets)

        if all_integer_addressable:
            return

        self._string_index_mode = True

        for dataset_idx, dataset in enumerate(self.datasets):
            if self._dataset_supports_int[dataset_idx]:
                continue
            keys = self._collect_string_keys(dataset)
            self._dataset_key_lists[dataset_idx] = keys
            self._register_dataset_keys(dataset_idx, keys)

    def _collect_string_keys(self, dataset):
        if isinstance(dataset, Mapping):
            keys_iter = dataset.keys()
        elif hasattr(dataset, "keys") and callable(getattr(dataset, "keys")):
            keys_iter = dataset.keys()
        else:
            try:
                keys_iter = iter(dataset)
            except TypeError as err:
                raise TypeError(
                    "Datasets with string indices must be iterable to expose keys."
                ) from err

        keys = list(keys_iter)
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(
                    "Datasets operating in string-index mode must provide string keys."
                )
        return keys

    def _register_dataset_keys(self, dataset_idx: int, keys: List[str]):
        for key in keys:
            if key in self._string_key_to_dataset:
                raise ValueError(
                    f"Duplicate utterance ID '{key}' detected across datasets."
                )
            self._string_key_to_dataset[key] = (dataset_idx, key)

    def _resolve_global_index(self, idx: int) -> Tuple[int, int]:
        """Resolve a global index to ``(dataset_idx, index within dataset)``.

        Raises:
            IndexError: If ``idx`` is negative or out of range.
        """
        if idx < 0:
            raise IndexError("Index out of range in CombinedDataset")
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                ds_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return i, ds_idx
        raise IndexError("Index out of range in CombinedDataset")

    def get_uid(self, idx: int) -> str:
        """Return the dataset-hash UID for a global integer index.

        This is the single source of truth for the UID used consistently
        from ``collect_stats`` (shape-file keys) through batching. See the
        class docstring's "Dataset-hash UID protocol" section.

        Args:
            idx (int): Global index into the combined dataset
                (``0 <= idx < len(self)``).

        Returns:
            str: The UID for this index -- ``str(idx)`` when this instance
            was built without ``uid_prefixes`` (backward compatibility);
            otherwise ``"<hash>:<position>"``, or (only for an unshardable
            shard whose ``shard()`` result exposes no ``indices``)
            ``"<hash>@s<shard_idx>:<shard-local position>"``, which is not a
            well-formed UID (``parse_uid`` returns ``None`` for it) and is
            only a unique label for the preprocessor/collator.

        Raises:
            IndexError: If ``idx`` is negative or out of range.
        """
        dataset_idx, ds_idx = self._resolve_global_index(idx)
        if self._uid_prefixes is None:
            return str(idx)
        position_map = self._position_maps[dataset_idx]
        if position_map is None:
            pos = ds_idx
        elif position_map is _UNRESOLVABLE:
            return f"{self._uid_prefixes[dataset_idx]}@s{self._shard_index}:{ds_idx}"
        else:
            pos = position_map[ds_idx]
        return format_uid(self._uid_prefixes[dataset_idx], pos)

    @property
    def uid_entries(self) -> Optional[List[DatasetUidEntry]]:
        """The ``DatasetUidEntry`` list this instance was built with, if any.

        ``None`` when built without ``uid_prefixes`` (backward compatibility);
        ``collect_stats`` then does not write a UID table and training does
        not validate against one.
        """
        return self._uid_entries

    def _select_reference_key_for_dataset(self, dataset_idx: int):
        if not self._string_index_mode or self._dataset_supports_int[dataset_idx]:
            return 0

        keys = self._dataset_key_lists[dataset_idx]
        if not keys:
            raise RuntimeError("Unable to locate reference key for dataset.")
        return keys[0]

    def _resolve_string_mode_index(self, idx):
        if isinstance(idx, int):
            dataset_idx, ds_idx = self._resolve_global_index(idx)
            if self._dataset_supports_int[dataset_idx]:
                dataset_key = ds_idx
            else:
                keys = self._dataset_key_lists[dataset_idx]
                if keys is None:
                    raise RuntimeError("String dataset keys are not initialized.")
                dataset_key = keys[ds_idx]
            uid = self.get_uid(idx)
            return uid, dataset_idx, dataset_key

        if isinstance(idx, str):
            try:
                dataset_idx, dataset_key = self._string_key_to_dataset[idx]
            except KeyError as err:
                raise ValueError(
                    f"Utterance ID '{idx}' is not supported by the underlying datasets."
                ) from err
            return idx, dataset_idx, dataset_key

        raise TypeError("Index must be an integer or string utterance ID.")

    def _getitem_string_mode(self, idx):
        uid, dataset_idx, dataset_key = self._resolve_string_mode_index(idx)
        dataset = self.datasets[dataset_idx]
        transform, preprocessor = self.transforms[dataset_idx]

        sample = dataset[dataset_key]
        transformed = transform(sample)
        if self.use_espnet_preprocessor:
            transformed = preprocessor(uid, transformed)
        else:
            transformed = preprocessor(transformed)

        if self.use_espnet_collator:
            return uid, transformed
        return transformed

    def shard(self, shard_idx: int):
        """Return a sharded version of the combined dataset.

        This is used when handling large datasets that are split into shards
        for efficiency and distributed processing (ESPnet multiple-iterator mode).
        All datasets must be subclasses of `espnet3.data.dataset.ShardedDataset`,
        and implement a `shard()` method.

        The returned dataset carries over ``use_espnet_collator`` from ``self``
        (it is not reset), and ``uid_prefixes``/``uid_entries`` (dataset-hash
        UIDs never change on sharding, only their resolution does). Per
        sub-dataset, ``get_uid`` on the shard resolves to the *original*
        (pre-shard) position when ``shard()`` returns an object exposing
        ``indices`` (e.g. ``torch.utils.data.Subset``, the shape recommended
        in the Dataset Sharding guide); otherwise ``get_uid`` returns a
        ``"<hash>@s<shard_idx>:<local>"`` label that is not a well-formed UID
        (see ``get_uid``) since there is no way to recover the original
        position without an ``indices``-bearing shard.

        Args:
            shard_idx (int): Index of the shard to retrieve.

        Returns:
            CombinedDataset: A new CombinedDataset containing the sharded datasets.

        Raises:
            RuntimeError: If any dataset does not support sharding.
        """
        if not all(isinstance(dataset, ShardedDataset) for dataset in self.datasets):
            raise RuntimeError(
                "All dataset should be the subclass of "
                "espnet3.components.data.dataset.ShardedDataset."
            )
        sharded_datasets = []
        position_maps: List[Optional[Sequence[int]]] = []
        for dataset in self.datasets:
            sharded = dataset.shard(shard_idx)
            if hasattr(sharded, "indices"):
                position_maps.append(sharded.indices)
            else:
                position_maps.append(_UNRESOLVABLE)
            sharded_datasets.append(sharded)
        result = CombinedDataset(
            sharded_datasets,
            self.transforms,
            self.use_espnet_preprocessor,
            uid_prefixes=self._uid_prefixes,
            uid_entries=self._uid_entries,
        )
        result._position_maps = position_maps
        result._shard_index = shard_idx
        result.use_espnet_collator = self.use_espnet_collator
        return result

    def __repr__(self) -> str:
        """Return a concise, inspectable summary of combined datasets."""
        entries = []
        for idx, (dataset, (transform, preprocessor)) in enumerate(
            zip(self.datasets, self.transforms)
        ):
            entries.append(
                f"{idx}: {build_qualified_name(dataset)}(len={len(dataset)}) "
                f"transform={build_callable_name(transform)} "
                f"preprocessor={build_callable_name(preprocessor)}"
            )
        datasets_desc = ", ".join(entries)
        return (
            f"{self.__class__.__name__}("
            f"total_len={len(self)}, "
            f"use_espnet_preprocessor={self.use_espnet_preprocessor}, "
            f"multiple_iterator={self.multiple_iterator}, "
            f"datasets=[{datasets_desc}]"
            f")"
        )


class DatasetWithTransform:
    """Lightweight wrapper for applying a transform function to dataset items.

    This class wraps a dataset and applies a user-defined transform followed by a
    preprocessor function. It also supports ESPnet-style UID handling, where the
    preprocessor receives both a UID and the sample.

    Args:
        dataset (Any): A dataset implementing `__getitem__` and `__len__`.
        transform (Callable): A function applied to each sample before preprocessor.
        preprocessor (Callable): A function applied after the transform.
            If `use_espnet_preprocessor` is True, it must accept `(uid, sample)`
            as arguments. Otherwise, it must accept a single `sample`.
        use_espnet_preprocessor (bool): Whether to include the UID when calling
            the preprocessor. Required for ESPnet's `AbsPreprocessor` compatibility.

    Example:
        >>> def transform(sample):
        ...     return {
        ...         "text": sample["text"].upper()
        ...     }
        >>>
        >>> def preprocess(uid, sample):
        ...     return {
        ...         "text": f"[uid={uid}] " + sample["text"]
        ...     }
        >>>
        >>> wrapped = DatasetWithTransform(
        ...     my_dataset,
        ...     transform,
        ...     preprocess,
        ...     use_espnet_preprocessor=True
        ... )
        >>> uid_sample = wrapped[0]
        >>> print(uid_sample["text"])
        [uid=0] HELLO

    Raises:
        TypeError: If `preprocessor` is not callable.
        TypeError: If `transform` is not callable.
    """

    def __init__(self, dataset, transform, preprocessor, use_espnet_preprocessor=False):
        """Initialize DatasetWithTransform."""
        if transform is None:
            transform = do_nothing
        assert callable(transform), "transform must be callable."
        if preprocessor is None:
            preprocessor = do_nothing
        assert callable(preprocessor), "preprocessor must be callable."
        self.dataset = dataset
        self.transform = transform
        self.preprocessor = preprocessor
        self.use_espnet_preprocessor = use_espnet_preprocessor

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Retrieve and process a sample by index."""
        sample = self.dataset[idx]
        transformed = self.transform(sample)  # apply transform
        if self.use_espnet_preprocessor:
            transformed = self.preprocessor(str(idx), transformed)
        else:
            transformed = self.preprocessor(transformed)
        return transformed

    def __call__(self, idx):
        """Alias for __getitem__ to allow callable access."""
        return self.__getitem__(idx)

    def __repr__(self) -> str:
        """Return a concise, inspectable summary of the wrapped dataset."""
        return (
            f"{self.__class__.__name__}("
            f"dataset={build_qualified_name(self.dataset)}"
            f"(len={len(self.dataset)}), "
            f"transform={build_callable_name(self.transform)}, "
            f"preprocessor={build_callable_name(self.preprocessor)}, "
            f"use_espnet_preprocessor={self.use_espnet_preprocessor}"
            f")"
        )


class ShardedDataset(ABC, Dataset):
    """Abstract base class for datasets that support sharding.

    This interface is used when datasets are split into shards for parallel or
    distributed data loading. Any dataset subclassing `ShardedDataset` must
    implement the `shard()` method.

    Attributes:
        total_shards (int): Total number of shards in the dataset.
        dist_world_size (int): Distributed world size used by this sharding
            scheme.

    Note:
        - This class is intended to be used with `CombinedDataset` in ESPnet.
        - All datasets combined must subclass `ShardedDataset` if sharding is used.

    Example:
        >>> class MyDataset(ShardedDataset):
        ...     def __init__(self):
        ...         self.total_shards = 8
        ...         self.dist_world_size = 4
        ...     def shard(self, idx):
        ...         return Subset(self, shard_indices[idx])

    """

    def shard(self, idx: int):
        """Return a new dataset shard corresponding to the given index.

        This method must be implemented by subclasses to return a subset of the data
        for sharded training or evaluation.

        Args:
            idx (int): The index of the shard to return.

        Returns:
            Dataset: A dataset instance representing the shard.

        Raises:
            NotImplementedError: Always in the base class. Must be overridden.
        """
        raise NotImplementedError(
            "Please implement `shard` function, "
            "which should return a `torch.utils.data.Dataset` object "
            "representing the shard corresponding to the given index."
        )
