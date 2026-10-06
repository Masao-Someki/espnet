"""Contract-checking code shared by what declares typed inputs and outputs.

The declaration types themselves (:class:`~espnet3.api.inference.Field`,
:class:`~espnet3.api.inference.Kind`) live in :mod:`espnet3.api.inference`;
this package holds only the checking code built on them, such as
:mod:`.metrics` for a metric's declared inputs/outputs and :mod:`.dataset`
for a dataset item's and a manifest's.
"""

from __future__ import annotations
