"""Models as a typed pair of pure functions.

A task never calls a linen module directly: it holds a `Model[ParamsT, InT,
OutT]`, so its input and output types are checked at every call site. Inputs
and outputs with more than one array are struct dataclasses (`FlowInput(x, t)`,
`PolicyOut(logits, value)`), never tuples or dicts.
"""

from typing import Protocol, cast

import attrs
import flax.linen as nn
import jax

from core.types import PRNGKey, Variables


class Model[ParamsT, InT, OutT](Protocol):
    def init(self, rng: PRNGKey, example: InT) -> ParamsT: ...

    def apply(self, params: ParamsT, x: InT, *, train: bool) -> OutT: ...

    def tabulate(self, example: InT, *, depth: int | None = 2) -> str: ...


@attrs.frozen(kw_only=True)
class LinenModel[InT, OutT]:
    """`Model[Variables, InT, OutT]` over a linen module.

    The module's `__call__` must take `(x: InT, *, train: bool) -> OutT`; the
    type parameters are the caller's promise that it does, since linen's own
    signatures are untyped.
    """

    module: nn.Module

    def init(self, rng: PRNGKey, example: InT) -> Variables:
        return self.module.init(rng, example, train=False)

    def apply(self, params: Variables, x: InT, *, train: bool) -> OutT:
        return cast(OutT, self.module.apply(params, x, train=train))

    def tabulate(self, example: InT, *, depth: int | None = 2) -> str:
        """Flax's per-layer table: shapes, parameter counts, forward and
        backward FLOPs (from XLA, per layer -- see `core.stats` for why the
        whole-step count is taken differently)."""
        return self.module.tabulate(
            jax.random.key(0),
            example,
            train=False,
            depth=depth,
            compute_flops=True,
            compute_vjp_flops=True,
            console_kwargs={"width": 160, "force_terminal": False},
        )
