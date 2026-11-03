"""Enforces the two-kinds-of-type rule over `core/` and `tasks/`.

Every class defined in these packages must be one of:

* `@attrs.frozen` -- static (configs, tasks, hooks, callbacks);
* a frozen dataclass, i.e. `@flax.struct.dataclass` -- traced (states, metrics);
* an `enum.Enum`;
* a `typing.Protocol`;
* a linen `nn.Module` (frozen dataclasses already; linen's own contract).

Anything else -- a plain class, a mutable dataclass, a NamedTuple, a TypedDict --
fails here.
"""

import dataclasses
import enum
import importlib
import inspect
import pathlib
import pkgutil
from typing import Protocol

import attrs
import flax.linen as nn
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = ("core", "tasks")


def _modules() -> list[str]:
    names = []
    for package in PACKAGES:
        for info in pkgutil.walk_packages([str(ROOT / package)], prefix=f"{package}."):
            if not info.name.endswith("_test") and not info.name.endswith(".main"):
                names.append(info.name)
    return sorted(names)


def _is_frozen_attrs(cls: type) -> bool:
    if not attrs.has(cls):
        return False
    # attrs installs this `__setattr__` on frozen classes and only on them.
    return getattr(cls.__setattr__, "__name__", "") == "_frozen_setattrs"


def _is_frozen_dataclass(cls: type) -> bool:
    params = getattr(cls, "__dataclass_params__", None)
    return dataclasses.is_dataclass(cls) and params is not None and params.frozen


def _is_protocol(cls: type) -> bool:
    return Protocol in cls.__mro__[1:] or getattr(cls, "_is_protocol", False)


def violation(cls: type) -> str | None:
    if issubclass(cls, enum.Enum) or issubclass(cls, nn.Module) or _is_protocol(cls):
        return None
    if _is_frozen_attrs(cls) or _is_frozen_dataclass(cls):
        return None
    return f"{cls.__module__}.{cls.__qualname__}"


@pytest.mark.parametrize("module_name", _modules())
def test_every_class_is_frozen(module_name: str) -> None:
    module = importlib.import_module(module_name)
    defined = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module_name
    ]
    bad = [name for cls in defined if (name := violation(cls)) is not None]
    assert not bad, f"not attrs-frozen / struct dataclass / Enum / Protocol: {bad}"


def test_checker_rejects_plain_classes() -> None:
    class Plain:
        pass

    @dataclasses.dataclass
    class Mutable:
        x: int

    @attrs.define
    class MutableAttrs:
        x: int

    @attrs.frozen
    class Frozen:
        x: int

    assert violation(Plain) is not None
    assert violation(Mutable) is not None
    assert violation(MutableAttrs) is not None
    assert violation(Frozen) is None
