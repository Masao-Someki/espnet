"""Checking a dataset's declared item fields and a builder's manifest columns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from espnet3.api.inference import KINDS, Field


class DatasetContractError(ValueError):
    """A dataset item, or a manifest row, does not match its declaration.

    Examples:
        >>> raise DatasetContractError("item lacks declared field 'text'")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: item lacks ...
    """


def require_declared(cls: type, attr: str) -> None:
    """Raise because ``cls`` has no ``attr`` contract declaration.

    Every dataset and builder must declare its item fields / manifest
    columns to be used; there is no undeclared-but-unchecked mode.

    Examples:
        >>> class Undeclared:
        ...     pass
        >>> require_declared(Undeclared, "fields")
        Traceback (most recent call last):
        espnet3.components.contract.dataset.DatasetContractError: Undeclared ...
    """
    raise DatasetContractError(
        f"{cls.__qualname__} does not declare {attr}; add a `{attr}` class "
        "attribute."
    )


def check_fields(cls: type, attr: str) -> None:
    """Raise ``TypeError`` unless ``cls.<attr>`` is a well-formed tuple of Field.

    Unlike ``check_declaration`` (which pairs an inputs/outputs attribute and
    orders required fields before optional ones), a dataset's ``fields`` or
    ``manifest_columns`` is one standalone tuple: just a tuple of ``Field``,
    non-empty, with distinct names.

    Examples:
        >>> class Good:
        ...     fields = (Field("speech", "audio"), Field("text", "text"))
        >>> check_fields(Good, "fields")
        >>> class Bad:
        ...     fields = (Field("text", "text"), Field("text", "text"))
        >>> check_fields(Bad, "fields")
        Traceback (most recent call last):
        TypeError: Bad.fields repeats a name: ['text', 'text']
    """
    fields = getattr(cls, attr, None)
    if not isinstance(fields, tuple) or not all(isinstance(f, Field) for f in fields):
        raise TypeError(f"{cls.__qualname__}.{attr} must be a tuple of Field")
    if not fields:
        raise TypeError(f"{cls.__qualname__}.{attr} must name at least one field")
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise TypeError(f"{cls.__qualname__}.{attr} repeats a name: {names}")


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
        class_fields: ``type(dataset).fields``, or ``None`` if undeclared.
        config_fields: ``DataOrganizer(fields=...)``, or ``None`` if omitted.
        class_name: The dataset class's name, for the error.

    Returns:
        ``class_fields`` when given, else ``config_fields``, else ``None``
        (meaning: undeclared, not checked).

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
        DatasetContractError: ``built_manifests()`` is non-empty but
            ``manifest_columns`` is undeclared; a manifest's first row has
            the wrong number of columns; or a ``path``-kind column's value
            is not an existing file.

    Examples:
        >>> import tempfile
        >>> class ExampleBuilder:
        ...     manifest_columns = (Field("utt_id", "text"), Field("text", "text"))
        ...     manifest_header = False
        ...     def built_manifests(self, **kwargs):
        ...         path = Path(tempfile.mkdtemp()) / "train.tsv"
        ...         _ = path.write_text("utt1" + chr(9) + "hello")
        ...         return {"train": path}
        >>> check_manifests(ExampleBuilder())
    """
    manifests = builder.built_manifests(**kwargs)
    if not manifests:
        return
    columns = getattr(type(builder), "manifest_columns", None)
    if columns is None:
        raise DatasetContractError(
            f"{type(builder).__name__} built manifests {sorted(manifests)} "
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


__all__ = [
    "DatasetContractError",
    "check_fields",
    "check_item",
    "check_manifests",
    "fields_from_config",
    "reconcile_fields",
    "require_declared",
]
