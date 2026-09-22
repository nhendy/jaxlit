"""Test fixture helpers."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest


def parameterized_fixture(
    fn: Callable[[], Sequence[Any] | Mapping[Any, Any]] | None = None,
    /,
    *,
    scope: str = "function",
    name: str | None = None,
) -> Any:
    """Turn a collection-returning function into a fixture parametrized over it.

    Each element becomes one value of the fixture, so a test requesting it runs
    once per element. Return a sequence and each value is id'd by its index;
    return a mapping and each value is id'd by its key::

        @parameterized_fixture
        def mlp_config() -> list[MLPConfig]:
            return [MLPConfig(...), MLPConfig(...)]
        # -> test_thing[0], test_thing[1]

        @parameterized_fixture
        def mlp_config() -> dict[str, MLPConfig]:
            return {"wide": MLPConfig(...), "deep": MLPConfig(...)}
        # -> test_thing[wide], test_thing[deep]

    Any fixture that depends on the result inherits the fan-out.

    In both cases the parametrization is over ``range(len(values))`` rather than
    over the values themselves, and the element is recovered inside the fixture
    via ``request.param_index``. That keeps the values out of the generated ids:
    for a sequence pytest's IdMaker renders an int param as its own literal, so
    the index falls out with no ``ids=`` at all; for a mapping the keys are
    handed over as ``ids=`` instead.
    """

    def deco(fn: Callable[[], Sequence[Any] | Mapping[Any, Any]]) -> Any:
        result = fn()
        if isinstance(result, Mapping):
            # Insertion order is the declaration order, so ids line up with values.
            ids: list[str] | None = [str(k) for k in result]
            values = list(result.values())
        else:
            ids, values = None, list(result)

        if not values:
            # params=range(0) makes pytest silently *skip* every dependent test
            # rather than fail, which hides a broken generator behind a green run.
            raise ValueError(
                f"@parameterized_fixture: {fn.__name__}() returned no values"
            )

        def _fixture(request: pytest.FixtureRequest) -> Any:
            return values[request.param_index]

        _fixture.__name__ = fn.__name__
        _fixture.__doc__ = fn.__doc__
        return pytest.fixture(
            params=range(len(values)), ids=ids, scope=scope, name=name or fn.__name__
        )(_fixture)

    return deco if fn is None else deco(fn)
