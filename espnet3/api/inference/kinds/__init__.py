"""The kinds a field can hold.

``AudioKind``, ``TextKind`` and ``SegmentsKind`` are built in. A new
modality - a conversation, video, a JSON document - is a :class:`Kind`
subclass passed directly as a :class:`~espnet3.api.inference.field.Field`'s
``kind``, from a system or from a recipe's own ``src/``;
:class:`~espnet3.api.inference.base.InferenceAPI` needs no change for it::

    from espnet3.api.inference import Field, InferenceAPI, Kind

    class Messages(Kind):
        def check(self, value, field, model, *, output):
            if not isinstance(value, list):
                raise TypeError(f"{field.name} must be a list of turns")
            return value

    class Inference(InferenceAPI):
        inputs = (Field(name="messages", kind=Messages),)
        ...
"""

from __future__ import annotations

from espnet3.api.inference.kinds.audio import Audio, AudioKind
from espnet3.api.inference.kinds.base import Kind
from espnet3.api.inference.kinds.segments import SegmentsKind
from espnet3.api.inference.kinds.text import TextKind

__all__ = [
    "Audio",
    "AudioKind",
    "Kind",
    "SegmentsKind",
    "TextKind",
]
