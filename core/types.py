"""Shared type aliases and empty pytree sentinels.

Two kinds of type exist in this framework, and which one a thing is follows from
whether it can be traced:

* `@attrs.frozen` -- static: configs, tasks, models, hooks, callbacks. Never a
  jit input; closed over instead.
* `@flax.struct.dataclass` -- traced: states, batches, metrics. Every array a
  step touches lives in one of these.

`core/purity_test.py` enforces it.
"""

from collections.abc import Callable
from typing import Any, cast

import jax
import numpy as np
from flax import struct

# Typed key from `jax.random.key`, never the raw uint32 `PRNGKey`.
type PRNGKey = jax.Array

# A 0-d array. Shapes are enforced with eincheck where they matter.
type Scalar = jax.Array

# A batch leaf: numpy when it leaves the host pipeline, jax.Array inside jit.
type HostOrDeviceArray = jax.Array | np.ndarray

# Linen's variable collections. Opaque to the framework: it only ever passes
# them between `Model.init`, `Model.apply` and optax, never indexes them.
type Variables = Any


@struct.dataclass
class NoBatch:
    """Batch type of a task that makes its own data on device (RL, synthetic).

    A pytree with no leaves, so stacking K of them and scanning over the stack
    costs nothing -- `lax.scan` is driven by `length` instead.
    """


@struct.dataclass
class NoExtra:
    """`TrainState.extra` for a task that needs nothing beyond params."""


def jit[**P, R](
    fn: Callable[P, R],
    *,
    in_shardings: Any = None,
    out_shardings: Any = None,
    donate_argnums: tuple[int, ...] = (),
) -> Callable[P, R]:
    """`jax.jit` that keeps `fn`'s signature for the type checker.

    jax types the call of a jitted function as returning `Any`, which would
    erase every type parameter flowing through the loop. Shardings left as None
    are not passed at all, so jax infers them from the arguments.
    """
    kwargs: dict[str, Any] = {}
    if in_shardings is not None:
        kwargs["in_shardings"] = in_shardings
    if out_shardings is not None:
        kwargs["out_shardings"] = out_shardings
    return cast(Callable[P, R], jax.jit(fn, donate_argnums=donate_argnums, **kwargs))
