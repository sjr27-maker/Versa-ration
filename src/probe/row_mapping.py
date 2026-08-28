"""Shared read-path safety net for every store's row-to-model mapping.

Pydantic v2's default `extra="ignore"` means `Model(**kwargs)` silently
drops any keyword that isn't one of the model's own fields — it does
NOT raise. That means constructing a model straight from `dict(row)`
against a `SELECT *`, on its own, is not actually "self-defending"
against a migration that added a column nobody wired up: the column's
value would just vanish on read, exactly the failure mode this module
exists to close (see the webui AttributeError this was written after —
`TurnDiagnostics` was missing a field a migration had already added).

`assert_row_consumed` is what makes `Model(**mapped)` an honest
guarantee instead of a coincidence: call it immediately before
construction, once any renaming/coercion this row needs has already
happened, so `mapped`'s keys are exactly what's about to be passed
through. Every store's row-mapping method uses this same check.
"""

from __future__ import annotations

from pydantic import BaseModel


def assert_row_consumed(model_cls: type[BaseModel], mapped: dict) -> None:
    """Raise if `mapped` contains a key that isn't a field on
    `model_cls`. Any column meant to be dropped (e.g. a join key used
    only for grouping) must be popped out of `mapped` by the caller
    before this runs — that's an explicit, visible decision, not a
    silent one.
    """
    unconsumed = set(mapped.keys()) - set(model_cls.model_fields.keys())
    if unconsumed:
        raise ValueError(
            f"{model_cls.__name__}: row column(s) {sorted(unconsumed)} have "
            "no corresponding model field — a migration likely added a "
            "column without updating the model or its store's read mapping"
        )
