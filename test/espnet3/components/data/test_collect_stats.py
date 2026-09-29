# tests/test_collect_stats.py
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Import the functions under test (adjust import path if your file/module path differs)
from espnet3.components.data.collect_stats import (
    _build_fingerprint,
    _build_model,
    _chunk_indices,
    _instantiate_dataset,
    _resolve_uid_and_sample,
    collect_stats,
    collect_stats_batch,
)
from espnet3.components.data.dataset import CombinedDataset, do_nothing

mp.set_start_method("fork", force=True)

# ===============================================================
# Test Case Summary for Collect Stats
# ===============================================================

# Normal Cases
# | Test Function Name                  | Description                          |
# |--------------------------------------|-------------------------------------|
# | test_collect_stats_local_basic       | Verifies that local mode aggregates |
# | test_collect_stats_local_basic       |  counts/sums correctly and       　 |
# |                                      | writes expected feature/statistics files    |
# | test_collect_stats_parallel_basic    | Verifies that parallel mode         |
# |                                      | (runner + shard outputs) matches local      |
# |                                      | aggregation logic and writes expected files |
# | test_collect_stats_entry_point_basic | Verifies top-level collect_stats() function |
# |                                      | works for different modes and produces      |
# |                                      | correct aggregated results                  |

TEST_MODEL_TARGET = "test.espnet3.components.data.test_collect_stats." "DummyModel"
TEST_ORGANIZER_TARGET = (
    "test.espnet3.components.data.test_collect_stats." "DummyOrganizer"
)
TEST_COLLATE_TARGET = "test.espnet3.components.data.test_collect_stats." "DummyCollate"


# -----------------------------
# Dummy components for testing
# -----------------------------


class DummyDataset:
    """Minimal dataset that returns (uid, sample_dict).

    Each item has a variable time length and fixed feature dim.
    """

    def __init__(self, n=10, base_len=3, dim=4, use_espnet_preprocessor=False):
        self.n = n
        self.base_len = base_len
        self.dim = dim
        # Precompute lengths to keep things deterministic
        self.lengths = [self.base_len + (i % 3) for i in range(self.n)]
        self.use_espnet_preprocessor = use_espnet_preprocessor

    def __len__(self):
        return self.n

    def get_uid(self, idx):
        """Return the stable uid this dataset uses, mirroring CombinedDataset."""
        return f"utt{idx:03d}"

    def __getitem__(self, idx):
        uid = self.get_uid(idx)
        T = self.lengths[idx]
        # Create a simple pattern to let us verify sums deterministically
        # Shape: [T, D]
        x = torch.full((T, self.dim), float(idx), dtype=torch.float32)
        if self.use_espnet_preprocessor:
            return uid, {"x": x, "length": T}
        else:
            return {"x": x, "length": T}


class DummyOrganizer:
    """Hydra-instantiable organizer that exposes .train and .valid datasets."""

    def __init__(
        self, n_train=8, n_valid=5, base_len=3, dim=4, use_espnet_preprocessor=False
    ):
        self.train = DummyDataset(
            n=n_train,
            base_len=base_len,
            dim=dim,
            use_espnet_preprocessor=use_espnet_preprocessor,
        )
        self.valid = DummyDataset(
            n=n_valid,
            base_len=base_len,
            dim=dim,
            use_espnet_preprocessor=use_espnet_preprocessor,
        )


class _StableUidRecipeDataset:
    """Recipe-style dataset opting into CombinedDataset's stable UID protocol.

    Implements ``get_utt_id`` only (the real opt-in contract); ``get_uid``,
    ``uids()`` and ``has_stable_uids`` are then provided by the real
    ``CombinedDataset`` that wraps it (see ``StableIdDataset`` below), not
    reimplemented here.
    """

    def __init__(self, n=6, base_len=3, dim=4, uid_offset=0):
        self.n = n
        self.base_len = base_len
        self.dim = dim
        self.uid_offset = uid_offset
        self.lengths = [self.base_len + (i % 3) for i in range(self.n)]

    def __len__(self):
        return self.n

    def get_utt_id(self, idx: int) -> str:
        """Return this item's stable utterance id."""
        return f"stable{idx + self.uid_offset:03d}"

    def __getitem__(self, idx):
        T = self.lengths[idx]
        x = torch.full((T, self.dim), float(idx), dtype=torch.float32)
        return {"x": x, "length": T}


def StableIdDataset(n=6, base_len=3, dim=4, uid_offset=0):
    """Build a real ``CombinedDataset`` wrapping a ``get_utt_id`` dataset.

    Used wherever a test needs to exercise the stable uid path (invariants
    3/5) through the actual ``CombinedDataset.get_uid``/``uids``/
    ``has_stable_uids`` implementation (owned by ashigaru1), not a dummy that
    only mimics its API.
    """
    return CombinedDataset(
        [
            _StableUidRecipeDataset(
                n=n, base_len=base_len, dim=dim, uid_offset=uid_offset
            )
        ],
        [(do_nothing, do_nothing)],
    )


class StableIdOrganizer:
    """Hydra-instantiable organizer exposing .train/.valid as real CombinedDataset."""

    def __init__(self, n_train=6, n_valid=0, base_len=3, dim=4, uid_offset=0):
        self.train = StableIdDataset(
            n=n_train, base_len=base_len, dim=dim, uid_offset=uid_offset
        )
        self.valid = StableIdDataset(
            n=n_valid, base_len=base_len, dim=dim, uid_offset=uid_offset
        )


class DummyCollate:
    """Collate that pads to max length and returns:

    (uids, {"x": [B, T, D], "lengths": [B]})
    """

    def __init__(self, int_pad_value: int = -1):
        self.pad = int_pad_value

    def __call__(self, items):
        uids = [u for (u, _) in items]
        seqs = [s["x"] for (_, s) in items]
        lengths = torch.tensor([s["length"] for (_, s) in items], dtype=torch.long)
        max_len = int(max([len(x) for x in seqs]))
        dim = int(seqs[0].shape[-1])
        B = len(seqs)
        x = torch.full((B, max_len, dim), self.pad, dtype=torch.float32)
        for i, seq in enumerate(seqs):
            T = seq.shape[0]
            x[i, :T] = seq
        return uids, {"x": x, "lengths": lengths}


class DummyModel:
    """Model with collect_feats(**batch) that returns torch tensors:

    - "mel": [B, T, D] (copied from input)
    - "mel_lengths": [B]
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def collect_feats(self, *, x: torch.Tensor, lengths: torch.Tensor):
        # Return the batch as "mel", scaled to let us test aggregation
        mel = (x * self.scale).to(self.device)
        mel_lengths = lengths.to(self.device)
        return {"mel": mel, "mel_lengths": mel_lengths}


class NoCollectModel:
    def to(self, device):
        return self

    def eval(self):
        return self


# ---------------------------------------
# Hydra configs for instantiate(...) calls
# ---------------------------------------


def make_model_cfg(scale: float = 1.0):
    # Use a direct class reference to avoid import path brittleness
    return OmegaConf.create(
        {
            "_target_": TEST_MODEL_TARGET,
            "scale": scale,
        }
    )


def make_dataset_cfg(
    n_train=8, n_valid=5, base_len=3, dim=4, use_espnet_preprocessor=False
):
    return OmegaConf.create(
        {
            "_target_": TEST_ORGANIZER_TARGET,
            "n_train": n_train,
            "n_valid": n_valid,
            "base_len": base_len,
            "dim": dim,
            "use_espnet_preprocessor": use_espnet_preprocessor,
        }
    )


TEST_STABLE_ORGANIZER_TARGET = (
    "test.espnet3.components.data.test_collect_stats.StableIdOrganizer"
)


def make_stable_dataset_cfg(n_train=6, n_valid=0, base_len=3, dim=4, uid_offset=0):
    """Build a dataset_config for the stable-uid organizer (invariants 3/5)."""
    return OmegaConf.create(
        {
            "_target_": TEST_STABLE_ORGANIZER_TARGET,
            "n_train": n_train,
            "n_valid": n_valid,
            "base_len": base_len,
            "dim": dim,
            "uid_offset": uid_offset,
        }
    )


def make_dataloader_cfg(use_custom_collate: bool = True):
    if use_custom_collate:
        return OmegaConf.create(
            {
                "train": {},
                "valid": {},
                "collate_fn": {
                    "_target_": TEST_COLLATE_TARGET,
                    "int_pad_value": -1,
                },
            }
        )
    else:
        # Fallback path to CommonCollateFn (not used here)
        return OmegaConf.create(
            {
                "train": {},
                "valid": {},
            }
        )


def make_parallel_cfg(n_workers=2):
    # Matches your parallel.py expectations
    return OmegaConf.create({"env": "local", "n_workers": n_workers, "options": {}})


# -----------------
# Helper assertions
# -----------------


def _expected_total_count(dataset):
    # Sum of all per-utterance time lengths
    return sum(dataset.lengths)


def _load_npz_counts(dirpath: Path, feat_key: str):
    p = dirpath / f"{feat_key}_stats.npz"
    assert p.exists(), f"Expected stats file not found: {p}"
    data = np.load(p)
    return data["count"], data["sum"], data["sum_square"]


# ------------
# The tests
# ------------
@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("use_parallel", [False, True])
@pytest.mark.parametrize("use_espnet_preprocessor", [False, True])
def test_collect_stats_local_basic(
    tmp_path: Path, use_parallel, use_espnet_preprocessor
):
    # Verify that local (non-parallel) path aggregates counts/sums correctly
    # and writes expected files.
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_dataset_cfg(
        n_train=6,
        n_valid=0,
        base_len=3,
        dim=4,
        use_espnet_preprocessor=use_espnet_preprocessor,
    )
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    par_cfg = make_parallel_cfg(n_workers=2) if use_parallel else None

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run "train" split locally
    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=par_cfg,
        write_collected_feats=True,
        batch_size=3,
    )

    mode_dir = out_dir / "train"
    assert (mode_dir / "stats_keys").exists(), "stats_keys not written"

    for k in ["mel", "mel_lengths"]:
        npz = mode_dir / f"{k}_stats.npz"
        assert npz.exists(), f"{npz} missing"
        scp = mode_dir / "collect_feats" / f"{k}.scp"
        assert scp.exists(), f"{scp} missing"

    ds = instantiate(ds_cfg).train
    total_count = _expected_total_count(ds)
    cnt, s, sq = _load_npz_counts(mode_dir, "mel")
    assert int(cnt) == total_count, "Total frame count mismatch for mel"


def test_build_model_rejects_missing_collect_feats():
    cfg = OmegaConf.create(
        {
            "model_config": {"_target_": f"{__name__}.NoCollectModel"},
        }
    )

    with pytest.raises(
        AttributeError, match="missing required callable 'collect_feats'"
    ):
        _build_model(cfg)


def test_collect_stats_batch_rejects_invalid_collate_shape():
    dataset = DummyDataset(n=2)
    model = DummyModel()

    def bad_collate(_items):
        return {"x": torch.zeros(1)}

    with pytest.raises(RuntimeError, match="return \\(uids, batch_dict\\)"):
        collect_stats_batch(
            [0, 1],
            model=model,
            dataset=dataset,
            collate_fn=bad_collate,
            device=torch.device("cpu"),
        )


def test_collect_stats_batch_rejects_non_mapping_features():
    dataset = DummyDataset(n=2)
    model = DummyModel()

    def bad_collate(_items):
        return ["utt0", "utt1"], ["not", "a", "mapping"]

    with pytest.raises(RuntimeError, match="return a mapping for batch tensors"):
        collect_stats_batch(
            [0, 1],
            model=model,
            dataset=dataset,
            collate_fn=bad_collate,
            device=torch.device("cpu"),
        )


def test_collect_stats_batch_rejects_conflicting_kwargs():
    dataset = DummyDataset(n=2)
    model = DummyModel()
    collate = DummyCollate()

    with pytest.raises(ValueError, match="kwargs conflict with batch tensors"):
        collect_stats_batch(
            [0, 1],
            model=model,
            dataset=dataset,
            collate_fn=collate,
            device=torch.device("cpu"),
            collect_stats_kwargs={"x": torch.zeros(1)},
        )


def test_chunk_indices_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        _chunk_indices(10, 0)


def test_instantiate_dataset_rejects_missing_split():
    dataset_cfg = make_dataset_cfg(n_train=1, n_valid=0)

    with pytest.raises(ValueError, match="does not provide split 'test'"):
        _instantiate_dataset(dataset_cfg, "test")


# ----------------------------
# Entry-point level smoke tests
# ----------------------------


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("use_parallel", [False, True])
def test_collect_stats_entrypoint_train(tmp_path: Path, use_parallel):
    # Smoke-test the public entrypoint `collect_stats` for the 'train' split,
    # both in local (no parallel_config) and parallel (with parallel_config) modes.
    # erifies that stats are saved and key files exist.
    # Also checks that *_lengths features are persisted (since they matter).
    model_cfg = make_model_cfg(scale=1.5)
    ds_cfg = make_dataset_cfg(n_train=6, n_valid=0, base_len=3, dim=4)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    par_cfg = make_parallel_cfg(n_workers=2) if use_parallel else None

    out_dir = tmp_path / "out_ep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Call the public entrypoint
    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=par_cfg,
        write_collected_feats=True,
        batch_size=2,
    )

    mode_dir = out_dir / "train"

    # Basic files
    assert (mode_dir / "stats_keys").exists(), "stats_keys not written by entrypoint"
    # Stats exist for both mel and mel_lengths
    for k in ["mel", "mel_lengths"]:
        npz = mode_dir / f"{k}_stats.npz"
        assert npz.exists(), f"{npz} not written by entrypoint"

    # Feature scps exist (including lengths)
    for k in ["mel", "mel_lengths"]:
        scp = mode_dir / "collect_feats" / f"{k}.scp"
        assert scp.exists(), f"{scp} not written by entrypoint"

    # Count check matches dataset total frames for mel
    ds = instantiate(ds_cfg).train
    total_count = _expected_total_count(ds)
    cnt, s, sq = _load_npz_counts(mode_dir, "mel")
    assert int(cnt) == total_count, "Total frame count mismatch in entrypoint(train)"


@pytest.mark.execution_timeout(30)
@pytest.mark.parametrize("use_parallel", [False, True])
def test_collect_stats_entrypoint_valid(tmp_path: Path, use_parallel):
    # Same as above but for 'valid' split, to ensure both branches work.
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_dataset_cfg(n_train=0, n_valid=5, base_len=2, dim=3)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    par_cfg = make_parallel_cfg(n_workers=2) if use_parallel else None

    out_dir = tmp_path / "out_ep_valid"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="valid",
        output_dir=out_dir,
        task=None,
        parallel_config=par_cfg,
        write_collected_feats=True,
        batch_size=2,
    )

    mode_dir = out_dir / "valid"
    assert (mode_dir / "stats_keys").exists()
    for k in ["mel", "mel_lengths"]:
        assert (mode_dir / f"{k}_stats.npz").exists()
        assert (mode_dir / "collect_feats" / f"{k}.scp").exists()


@pytest.mark.parametrize("flag", [True, False])
def test_collect_stats_rejects_multiple_iterator(tmp_path: Path, flag):
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_dataset_cfg(n_train=4, n_valid=0, base_len=3, dim=4)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    dl_cfg.train.multiple_iterator = flag

    with pytest.raises(RuntimeError, match="multiple_iterator"):
        collect_stats(
            model_config=model_cfg,
            dataset_config=ds_cfg,
            dataloader_config=dl_cfg,
            mode="train",
            output_dir=tmp_path / "out_reject",
            task=None,
            parallel_config=None,
            write_collected_feats=False,
            batch_size=2,
        )


# =====================================================================
# Invariant 3: _resolve_uid_and_sample -- uid/sample resolution without
# guessing tuple-shape from a separate flag (T4)
# =====================================================================


def test_resolve_uid_and_sample_uses_get_uid_for_plain_sample():
    dataset = StableIdDataset(n=2)

    uid, sample = _resolve_uid_and_sample(dataset, 1)
    expected = dataset[1]

    assert uid == "stable001"
    assert sample["length"] == expected["length"]
    assert torch.equal(sample["x"], expected["x"])


def test_resolve_uid_and_sample_uses_get_uid_for_matching_tuple():
    dataset = DummyDataset(n=3, use_espnet_preprocessor=True)

    uid, sample = _resolve_uid_and_sample(dataset, 2)
    _, expected = dataset[2]

    assert uid == "utt002"
    assert sample["length"] == expected["length"]
    assert torch.equal(sample["x"], expected["x"])


def test_resolve_uid_and_sample_falls_back_to_str_index_without_get_uid():
    class NoUidDataset:
        def __getitem__(self, idx):
            return {"x": torch.full((2, 1), float(idx)), "length": 2}

    dataset = NoUidDataset()
    assert not hasattr(dataset, "get_uid")

    uid, sample = _resolve_uid_and_sample(dataset, 0)
    expected = dataset[0]

    assert uid == "0"
    assert sample["length"] == expected["length"]
    assert torch.equal(sample["x"], expected["x"])


def test_resolve_uid_and_sample_raises_on_uid_mismatch():
    class MismatchedDataset:
        def get_uid(self, idx):
            return f"authoritative{idx}"

        def __getitem__(self, idx):
            return ("different_uid", {"x": torch.zeros(1)})

    with pytest.raises(RuntimeError, match="does not match"):
        _resolve_uid_and_sample(MismatchedDataset(), 0)


@pytest.mark.parametrize("use_espnet_preprocessor", [False, True])
def test_collect_stats_batch_all_collator_preprocessor_combinations(
    use_espnet_preprocessor,
):
    """T4: every collator/preprocessor combination resolves uid via get_uid."""
    dataset = DummyDataset(
        n=4, base_len=2, dim=3, use_espnet_preprocessor=use_espnet_preprocessor
    )
    model = DummyModel(scale=1.0)
    collate = DummyCollate()

    stats, shape_info = collect_stats_batch(
        [0, 1, 2],
        model=model,
        dataset=dataset,
        collate_fn=collate,
        device=torch.device("cpu"),
    )

    expected_uids = {dataset.get_uid(i) for i in [0, 1, 2]}
    assert set(shape_info["mel"].keys()) == expected_uids


def test_collect_stats_batch_uid_matches_dataset_get_uid():
    """T2 (collect_stats side): shape_info keys equal dataset.get_uid(i)."""
    dataset = StableIdDataset(n=4, base_len=2, dim=3)
    model = DummyModel(scale=1.0)
    collate = DummyCollate()
    idxs = [0, 2, 3]

    stats, shape_info = collect_stats_batch(
        idxs,
        model=model,
        dataset=dataset,
        collate_fn=collate,
        device=torch.device("cpu"),
    )

    assert set(shape_info["mel"].keys()) == {dataset.get_uid(i) for i in idxs}


def test_collect_stats_batch_rejects_uid_mismatch():
    class MismatchedDataset:
        def get_uid(self, idx):
            return f"authoritative{idx}"

        def __getitem__(self, idx):
            return ("different_uid", {"x": torch.zeros(2, 1), "length": 2})

    with pytest.raises(RuntimeError, match="does not match"):
        collect_stats_batch(
            [0],
            model=DummyModel(),
            dataset=MismatchedDataset(),
            collate_fn=DummyCollate(),
            device=torch.device("cpu"),
        )


@pytest.mark.execution_timeout(30)
def test_collect_stats_writes_stable_uids_to_shape_file(tmp_path: Path):
    """T2 (collect_stats side): shape file keys == dataset.uids()."""
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_stable_dataset_cfg(n_train=5, n_valid=0, base_len=2, dim=3)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)

    out_dir = tmp_path / "out_stable"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=None,
        write_collected_feats=False,
        batch_size=2,
    )

    dataset = instantiate(ds_cfg).train
    shape_path = out_dir / "train" / "mel_shape"
    lines = shape_path.read_text(encoding="utf-8").splitlines()
    written_uids = {line.split(" ", 1)[0] for line in lines}

    assert written_uids == set(dataset.uids())
    assert len(lines) == len(dataset)


# =====================================================================
# Invariant 5: collect_stats resume rejects a stale fingerprint (T5)
# =====================================================================


def test_build_fingerprint_is_deterministic():
    dataset = StableIdDataset(n=3)
    model_cfg = make_model_cfg(scale=1.0)
    ds_cfg = make_stable_dataset_cfg(n_train=3)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)

    fp1 = _build_fingerprint(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        batch_size=2,
        write_collected_feats=False,
        dataset=dataset,
    )
    fp2 = _build_fingerprint(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        batch_size=2,
        write_collected_feats=False,
        dataset=dataset,
    )

    assert fp1 == fp2
    assert fp1["stable_uids"] is True
    assert fp1["num_items"] == 3


def test_build_fingerprint_changes_with_model_config():
    dataset = StableIdDataset(n=3)
    ds_cfg = make_stable_dataset_cfg(n_train=3)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)

    fp_a = _build_fingerprint(
        model_config=make_model_cfg(scale=1.0),
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        batch_size=2,
        write_collected_feats=False,
        dataset=dataset,
    )
    fp_b = _build_fingerprint(
        model_config=make_model_cfg(scale=2.0),
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        batch_size=2,
        write_collected_feats=False,
        dataset=dataset,
    )

    assert fp_a["sha256"] != fp_b["sha256"]


def test_build_fingerprint_uses_uids_hash_for_stable_dataset():
    ds_cfg = make_stable_dataset_cfg(n_train=3)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    model_cfg = make_model_cfg(scale=1.0)

    fp_no_stable = _build_fingerprint(
        model_config=model_cfg,
        dataset_config=make_dataset_cfg(n_train=3, n_valid=0),
        dataloader_config=dl_cfg,
        mode="train",
        batch_size=2,
        write_collected_feats=False,
        dataset=DummyDataset(n=3),
    )
    fp_stable = _build_fingerprint(
        model_config=model_cfg,
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        batch_size=2,
        write_collected_feats=False,
        dataset=StableIdDataset(n=3),
    )

    assert fp_no_stable["stable_uids"] is False
    assert fp_stable["stable_uids"] is True


@pytest.mark.execution_timeout(30)
def test_collect_stats_rerun_with_changed_model_raises(tmp_path: Path):
    ds_cfg = make_dataset_cfg(n_train=4, n_valid=0, base_len=3, dim=4)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    out_dir = tmp_path / "out_resume_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=make_model_cfg(scale=1.0),
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=None,
        write_collected_feats=False,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="fingerprint"):
        collect_stats(
            model_config=make_model_cfg(scale=2.0),
            dataset_config=ds_cfg,
            dataloader_config=dl_cfg,
            mode="train",
            output_dir=out_dir,
            task=None,
            parallel_config=None,
            write_collected_feats=False,
            batch_size=2,
        )


@pytest.mark.execution_timeout(30)
def test_collect_stats_rerun_with_changed_dataset_raises(tmp_path: Path):
    model_cfg = make_model_cfg(scale=1.0)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    out_dir = tmp_path / "out_resume_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=model_cfg,
        dataset_config=make_dataset_cfg(n_train=3, n_valid=0, base_len=3, dim=4),
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=None,
        write_collected_feats=False,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="fingerprint"):
        collect_stats(
            model_config=model_cfg,
            dataset_config=make_dataset_cfg(n_train=5, n_valid=0, base_len=3, dim=4),
            dataloader_config=dl_cfg,
            mode="train",
            output_dir=out_dir,
            task=None,
            parallel_config=None,
            write_collected_feats=False,
            batch_size=2,
        )


@pytest.mark.execution_timeout(30)
def test_collect_stats_rerun_with_same_length_but_different_uids_raises(
    tmp_path: Path,
):
    """T5 (optional improvement): same num_items, but the dataset's stable
    uid content changed (e.g. a source file reorder) -- the uids_sha256 in
    the fingerprint must still catch this even though num_items and every
    other config field are identical."""
    model_cfg = make_model_cfg(scale=1.0)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    out_dir = tmp_path / "out_resume_same_len_diff_uids"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=model_cfg,
        dataset_config=make_stable_dataset_cfg(n_train=4, n_valid=0, uid_offset=0),
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=None,
        write_collected_feats=False,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="fingerprint"):
        # Same n_train (same num_items) but a different uid_offset shifts
        # every stable uid -- content changed, length did not.
        collect_stats(
            model_config=model_cfg,
            dataset_config=make_stable_dataset_cfg(
                n_train=4, n_valid=0, uid_offset=100
            ),
            dataloader_config=dl_cfg,
            mode="train",
            output_dir=out_dir,
            task=None,
            parallel_config=None,
            write_collected_feats=False,
            batch_size=2,
        )


@pytest.mark.execution_timeout(30)
def test_collect_stats_resume_false_recomputes(tmp_path: Path):
    ds_cfg = make_dataset_cfg(n_train=3, n_valid=0, base_len=3, dim=4)
    dl_cfg = make_dataloader_cfg(use_custom_collate=True)
    out_dir = tmp_path / "out_resume_false"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_stats(
        model_config=make_model_cfg(scale=1.0),
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=None,
        write_collected_feats=False,
        batch_size=2,
    )

    # Changed model + resume=False must recompute instead of raising.
    collect_stats(
        model_config=make_model_cfg(scale=2.0),
        dataset_config=ds_cfg,
        dataloader_config=dl_cfg,
        mode="train",
        output_dir=out_dir,
        task=None,
        parallel_config=None,
        write_collected_feats=False,
        batch_size=2,
        resume=False,
    )

    mode_dir = out_dir / "train"
    cnt, s, sq = _load_npz_counts(mode_dir, "mel")
    ds = instantiate(ds_cfg).train
    expected_sum = sum(
        2.0 * idx * length * ds.dim for idx, length in enumerate(ds.lengths)
    )
    assert int(cnt) == _expected_total_count(ds)
    # scale=2.0 run must have actually overwritten the scale=1.0 stats file.
    assert float(s.sum()) == pytest.approx(expected_sum)
