"""Checking a dataset's declared item fields and a builder's manifest columns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from espnet3.api.inference import KINDS, Field


class DatasetContractError(ValueError):
    """A dataset item, a manifest row, or a metric input does not match its declaration.

    Examples:
        >>> raise DatasetContractError("item lacks declared field 'text'")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: item lacks ...
    """


def require_fields(obj, attr: str) -> None:
    """Raise because ``obj`` declares no ``attr``.

    Every dataset and builder must declare its item fields / manifest
    columns - as a class attribute, or set on ``self`` for one that
    depends on its own configuration; there is no undeclared fallback.

    Args:
        obj: The class to check, or an instance whose own attributes
            (set in its ``__init__``) declare a contract its class does
            not.
        attr: The attribute name, such as ``"fields"``.

    Examples:
        >>> class Undeclared:
        ...     pass
        >>> require_fields(Undeclared, "fields")
        Traceback (most recent call last):
        TypeError: Undeclared does not declare fields; declare `fields` ...
    """
    cls = obj if isinstance(obj, type) else type(obj)
    raise TypeError(
        f"{cls.__qualname__} does not declare {attr}; declare `{attr}` "
        "(a class attribute, or set on `self`)."
    )


def check_fields(obj, attr: str) -> Optional[Tuple[Field, ...]]:
    """Return ``obj.<attr>`` if well-formed, or ``None`` if undeclared.

    Unlike ``check_declaration`` (which pairs an inputs/outputs attribute
    and orders required fields before optional ones), a dataset's
    ``fields`` or a builder's ``manifest_columns`` is one standalone
    tuple: just a tuple of ``Field``, non-empty, with distinct names.
    ``None`` lets a caller decide whether an undeclared ``attr`` is
    allowed (a builder with no manifest of its own) or not (call
    :func:`require_fields`).

    Args:
        obj: The class to check, or an instance whose own attributes
            (set in its ``__init__``) declare a contract its class does
            not.
        attr: The attribute name, such as ``"fields"``.

    Raises:
        TypeError: ``obj.<attr>`` is declared but malformed.

    Examples:
        >>> class Good:
        ...     fields = (Field("speech", "audio"), Field("text", "text"))
        >>> check_fields(Good, "fields")[0].name
        'speech'
        >>> class Bad:
        ...     fields = (Field("text", "text"), Field("text", "text"))
        >>> check_fields(Bad, "fields")
        Traceback (most recent call last):
        TypeError: Bad.fields repeats a name: ['text', 'text']
    """
    cls = obj if isinstance(obj, type) else type(obj)
    fields = getattr(obj, attr, None)
    if fields is None:
        return None
    if not isinstance(fields, tuple) or not all(isinstance(f, Field) for f in fields):
        raise TypeError(f"{cls.__qualname__}.{attr} must be a tuple of Field")
    if not fields:
        raise TypeError(f"{cls.__qualname__}.{attr} must name at least one field")
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise TypeError(f"{cls.__qualname__}.{attr} repeats a name: {names}")
    return fields


def fields_from_config(mapping: Mapping[str, str]) -> Tuple[Field, ...]:
    """Turn a config's ``{name: kind}`` mapping into a tuple of ``Field``.

    Examples:
        >>> fields = fields_from_config({"speech": "audio", "text": "text"})
        >>> [(f.name, f.kind) for f in fields]
        [('speech', 'audio'), ('text', 'text')]
    """
    return tuple(Field(name, kind) for name, kind in mapping.items())


def _format_fields(fields: Tuple[Field, ...], *, brackets: str) -> str:
    inner = ", ".join(f"{f.name}: {f.kind}" for f in fields)
    left, right = brackets
    return f"{left}{inner}{right}"


def reconcile_fields(
    class_fields: Optional[Tuple[Field, ...]],
    config_fields: Optional[Tuple[Field, ...]],
    *,
    class_name: str,
) -> Optional[Tuple[Field, ...]]:
    """Return the effective item declaration, requiring agreement when both exist.

    Args:
        class_fields: The dataset's own declared ``fields``, from
            :func:`check_fields`, or ``None`` if undeclared.
        config_fields: ``DataOrganizer(fields=...)``, or ``None`` if omitted.
        class_name: The dataset class's name, for the error.

    Returns:
        ``class_fields`` when given, else ``config_fields``, else ``None``
        (meaning: undeclared; the caller must call :func:`require_fields`).

    Raises:
        DatasetContractError: Both are given and disagree on name or kind.

    Examples:
        >>> speech = (Field("speech", "audio"), Field("text", "text"))
        >>> reconcile_fields(speech, None, class_name="MyDataset") == speech
        True
        >>> reconcile_fields(None, speech, class_name="MyDataset") == speech
        True
        >>> reconcile_fields(speech, (Field("text", "text"),), class_name="MyDataset")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: dataset.fields ...
    """
    if class_fields is not None and config_fields is not None:
        if {(f.name, f.kind) for f in class_fields} != {
            (f.name, f.kind) for f in config_fields
        }:
            raise DatasetContractError(
                "dataset.fields in the config declares "
                f"{_format_fields(config_fields, brackets='{}')} but "
                f"{class_name}.fields declares "
                f"{_format_fields(class_fields, brackets='()')}; "
                "make them agree or drop one"
            )
    return class_fields if class_fields is not None else config_fields


def check_item(fields: Tuple[Field, ...], item: Any, where: str) -> None:
    """Raise unless ``item`` has every declared field, each holding its kind.

    Args:
        fields: The declared item fields.
        item: One dataset sample, taken after its transform and before its
            preprocessor.
        where: Where ``item`` came from, for the error.

    Raises:
        DatasetContractError: ``item`` is not a mapping, lacks a required
            declared field, or a field's value does not match its kind. An
            undeclared extra key is not an error.

    Examples:
        >>> import numpy as np
        >>> fields = (Field("speech", "audio"), Field("text", "text"))
        >>> item = {"speech": np.zeros(16000, dtype=np.float32), "text": "hi"}
        >>> check_item(fields, item, "train[0]")
        >>> check_item(fields, {"speech": item["speech"]}, "train[0]")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: train[0]: ...
    """
    if not isinstance(item, Mapping):
        raise DatasetContractError(
            f"{where}: item must be a dict, got {type(item).__name__}"
        )
    for f in fields:
        if f.name not in item:
            if f.optional:
                continue
            raise DatasetContractError(
                f"{where}: item lacks declared field {f.name!r} ({f.kind}); "
                f"keys are {sorted(item)}"
            )
        if not KINDS[f.kind].accepts(item[f.name], f):
            raise DatasetContractError(
                f"{where}: field {f.name!r} is declared {f.kind} but the item "
                f"holds {type(item[f.name]).__name__}"
            )


def check_declared_fields(
    dataset: Any, config_fields: Optional[Tuple[Field, ...]]
) -> Tuple[Field, ...]:
    """Return ``dataset``'s effective declared fields, or raise.

    Combines :func:`check_fields` (the dataset's own declaration,
    structurally valid or ``None``), :func:`reconcile_fields` (agreement
    with ``config_fields``), and :func:`require_fields` (neither is
    given): the single call every use site - ``CombinedDataset``,
    ``DataOrganizer``'s standalone test datasets - makes before
    :func:`check_item`, so a dataset with no reachable declaration can
    never be used unchecked.

    Args:
        dataset: A dataset instance (its class, or its own ``__init__``,
            may declare ``fields``).
        config_fields: ``DataOrganizer(fields=...)``, or ``None``.

    Returns:
        The effective declared fields.

    Raises:
        TypeError: ``dataset.fields`` is declared but malformed, or
            neither it nor ``config_fields`` is given.
        DatasetContractError: Both are given and disagree.
    """
    class_fields = check_fields(dataset, "fields")
    declared = reconcile_fields(
        class_fields, config_fields, class_name=type(dataset).__qualname__
    )
    if declared is None:
        require_fields(dataset, "fields")
    return declared


def _first_row(path: Path, *, header: bool) -> List[str]:
    """Return the first data row of a tab-separated manifest, as a list of cells."""
    with open(path, encoding="utf-8") as f:
        if header:
            next(f, None)
        line = next(f, "")
    return line.rstrip("\n").split("\t")


def check_manifests(builder: Any, **kwargs) -> None:
    """Raise unless each built manifest's first row matches its declared columns.

    A builder whose ``built_manifests()`` is empty (it writes no espnet3
    manifest of its own) is not checked; one that does write manifests
    must declare ``manifest_columns`` to be used.

    Args:
        builder: A ``DatasetBuilder`` instance, already built.
        **kwargs: Passed to ``builder.built_manifests(**kwargs)``.

    Raises:
        TypeError: ``manifest_columns`` is declared but malformed.
        DatasetContractError: ``built_manifests()`` is non-empty but
            ``manifest_columns`` is undeclared; a manifest's first row has
            the wrong number of columns; or a ``path``-kind column's value
            is not an existing file.

    Examples:
        A fixed declaration (class attribute):

        >>> import tempfile
        >>> class ExampleBuilder:
        ...     manifest_columns = (Field("utt_id", "text"), Field("text", "text"))
        ...     manifest_header = False
        ...     def built_manifests(self, **kwargs):
        ...         path = Path(tempfile.mkdtemp()) / "train.tsv"
        ...         _ = path.write_text("utt1" + chr(9) + "hello")
        ...         return {"train": path}
        >>> check_manifests(ExampleBuilder())

        A declaration the builder's own ``__init__`` builds:

        >>> class ConfigurableBuilder:
        ...     def __init__(self, extra_column):
        ...         self.manifest_columns = (
        ...             Field("utt_id", "text"), Field(extra_column, "text")
        ...         )
        ...         self.manifest_header = False
        ...     def built_manifests(self, **kwargs):
        ...         path = Path(tempfile.mkdtemp()) / "train.tsv"
        ...         _ = path.write_text("utt1" + chr(9) + "hello")
        ...         return {"train": path}
        >>> check_manifests(ConfigurableBuilder("speaker"))
    """
    manifests = builder.built_manifests(**kwargs)
    if not manifests:
        return
    columns = check_fields(builder, "manifest_columns")
    if columns is None:
        raise DatasetContractError(
            f"{type(builder).__qualname__} built manifests {sorted(manifests)} "
            "but does not declare manifest_columns; add one to check them."
        )
    header = getattr(builder, "manifest_header", False)
    for split, path in manifests.items():
        row = _first_row(Path(path), header=header)
        if len(row) != len(columns):
            raise DatasetContractError(
                f"manifest {path} ({split}) row 1 has {len(row)} columns, "
                f"{type(builder).__name__} declares {len(columns)}: "
                f"{[c.name for c in columns]}"
            )
        for col, value in zip(columns, row):
            if col.kind == "path" and not Path(value).exists():
                raise DatasetContractError(
                    f"manifest {path} ({split}) column {col.name!r} points to "
                    f"a missing file: {value}"
                )


def check_dataset_column_kind(
    fields: Optional[Tuple[Field, ...]], column: str, wanted_kind: str, *, where: str
) -> None:
    """Raise unless a dataset's declared ``column`` has kind ``wanted_kind``.

    For a metric input sourced from the test set itself
    (``dataset:<column>``), checked once the dataset's declared fields
    are known. There is no silent skip: an undeclared dataset, or one
    whose declared fields do not name ``column``, raises too.

    Args:
        fields: The dataset's declared item fields (see
            :func:`check_declared_fields`).
        column: The dataset column a metric input reads.
        wanted_kind: The kind the metric's declared input expects.
        where: What is being checked, for the error.

    Raises:
        DatasetContractError: ``fields`` is empty, does not name
            ``column``, or names it with a different kind than
            ``wanted_kind``.

    Examples:
        >>> fields = (Field("speech", "audio"), Field("text", "text"))
        >>> check_dataset_column_kind(fields, "text", "text", where="x")
        >>> check_dataset_column_kind(fields, "text", "audio", where="x")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: x: dataset ...
    """
    if not fields:
        raise DatasetContractError(
            f"{where}: the dataset declares no fields; cannot check "
            f"column {column!r}"
        )
    match = next((f for f in fields if f.name == column), None)
    if match is None:
        raise DatasetContractError(
            f"{where}: the dataset's declared fields do not name " f"column {column!r}"
        )
    if match.kind != wanted_kind:
        raise DatasetContractError(
            f"{where}: dataset column {column!r} is declared {match.kind!r} "
            f"but the metric input wants {wanted_kind!r}"
        )


__all__ = [
    "DatasetContractError",
    "check_declared_fields",
    "check_dataset_column_kind",
    "check_fields",
    "check_item",
    "check_manifests",
    "fields_from_config",
    "reconcile_fields",
    "require_fields",
]
