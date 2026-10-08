"""The declaration of one input or output of a system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from espnet3.api.inference.kinds.base import Kind


@dataclass(frozen=True, kw_only=True)
class Field:
    """One input or output of a system.

    A system lists these in its ``inputs`` and ``outputs``; the base class
    binds arguments to them, converts and checks values by ``kind``, and a
    front end builds its widgets or command-line arguments from them.

    Args:
        name: The keyword the value is passed or returned as, such as
            ``speech`` or ``text``. Must be a Python identifier.
        kind: A :class:`Kind` subclass or instance. Decides how a value is
            converted and checked (``AudioKind`` becomes an :class:`Audio`
            at the model's rate, ``TextKind`` must be a ``str``) and, for a
            front end, which widget or argument type shows it. A subclass
            is instantiated.
        label: What a page calls the field. Defaults to the name with
            underscores spaced and the first letter capitalised.
        optional: An input the caller may leave out; the hook then does
            not receive it. Outputs are never optional.
        channels: For ``AudioKind`` only: how many channels the hook sees.
            ``1`` (the default) gives the reference channel as a 1-D
            array, what a single-channel backend takes; ``None`` gives
            every channel as ``(channels, samples)``; a count ``N`` demands
            exactly ``N``. Other kinds ignore it.

    Raises:
        TypeError: If ``kind`` is not a :class:`Kind` subclass or instance.
        ValueError: If ``name`` is not an identifier, or ``channels`` is
            below 1.

    Examples:
        >>> from espnet3.api.inference.kinds import AudioKind, TextKind
        >>> Field(name="speech", kind=AudioKind).kind   # a class is instantiated
        AudioKind()
        >>> Field(name="reference_speech", kind=AudioKind, optional=True).label
        'Reference speech'
        >>> Field(name="text", kind=TextKind, label="Transcription").label
        'Transcription'
        >>> mixture = Field(name="mixture", kind=AudioKind, channels=None)
        >>> mixture.channels   # a multichannel model
        >>> Field(name="speech", kind=AudioKind).channels  # the reference channel
        1
        >>> same_kind = Field(name="speech", kind=AudioKind())
        >>> Field(name="speech", kind=AudioKind) == same_kind
        True
    """

    name: str
    kind: Union[Kind, type[Kind]]
    label: str = ""
    optional: bool = False
    channels: Optional[int] = 1

    def __post_init__(self) -> None:
        """Instantiate a ``kind`` class, then check the kind and the name."""
        kind = self.kind
        if isinstance(kind, type) and issubclass(kind, Kind):
            kind = kind()
        if not isinstance(kind, Kind):
            raise TypeError(
                f"Field {self.name!r}: kind must be a Kind class or instance, "
                f"not {kind!r}"
            )
        object.__setattr__(self, "kind", kind)
        if not self.name.isidentifier():
            raise ValueError(f"Field name {self.name!r} must be a Python identifier")
        if self.channels is not None and self.channels < 1:
            raise ValueError(f"Field {self.name!r}: channels must be None or >= 1")
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").capitalize())
