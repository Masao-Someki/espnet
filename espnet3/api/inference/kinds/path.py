"""The ``path`` kind: a filesystem path, as ``str`` or ``os.PathLike``."""

from __future__ import annotations

import os

from espnet3.api.inference.kinds.base import Kind


class PathKind(Kind):
    """``path``: a filesystem path; used for manifest columns such as a wav path."""

    def check(self, value, field, model, *, output):
        """Require ``str`` or ``os.PathLike``; return it unchanged.

        Args:
            value: What the caller gave, or what the hook returned.
            field: The declaration, for the name in the error.
            model: Unused; a path needs nothing from the model.
            output: Whether ``value`` is a hook's result.

        Returns:
            ``value`` itself.

        Raises:
            TypeError: If ``value`` is not a ``str`` or ``os.PathLike``.

        Examples:
            >>> PathKind().check("a.wav", Field("wav", "path"), model, output=False)
            'a.wav'
            >>> PathKind().check(7, Field("wav", "path"), model, output=True)
            Traceback (most recent call last):
            TypeError: 'wav' returned as int, must be a str or os.PathLike
        """
        if not isinstance(value, (str, os.PathLike)):
            where = "returned" if output else "given"
            raise TypeError(
                f"{field.name!r} {where} as {type(value).__name__}, "
                "must be a str or os.PathLike"
            )
        return value
