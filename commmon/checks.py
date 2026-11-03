"""One process-wide switch for eincheck's runtime shape checks.

eincheck only offers context managers (`enable_checks()`/`disable_checks()`);
this holds one open for the life of the process, so an entry point can turn
checks off once instead of wrapping every call site.

Per *process*: Grain starts its workers with `spawn`, so each worker begins
with checks on whatever the parent set. `data.transforms.ShapeChecks` re-applies
the setting inside each worker.

Turning checks off is for throughput, not for skipping validation altogether:
ingest validates every record when it is written, so the per-example re-checks
that every `Scenario.replace()` triggers catch nothing new -- and cost ~80% of
the input pipeline's CPU time.
"""

import contextlib

import eincheck

_stack = contextlib.ExitStack()
_current: bool | None = None


def set_shape_checks(enabled: bool) -> None:
    """Turn eincheck on or off for this whole process. Cheap when unchanged."""
    global _current
    if _current is enabled:
        return
    _stack.close()
    _stack.enter_context(
        eincheck.enable_checks() if enabled else eincheck.disable_checks()
    )
    _current = enabled


def shape_checks_enabled() -> bool:
    """The current setting; eincheck's default, on, until `set_shape_checks`."""
    return True if _current is None else _current
