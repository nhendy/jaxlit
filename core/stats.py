"""Parameter counts and FLOPs of a task's train step.

FLOPs are counted from the step's jaxpr, not taken from XLA's cost analysis:
XLA counts a `while`/`scan` body once regardless of its trip count, so it
under-reports any step built on `lax.scan` -- the loop's own K-steps-per-call
scan, and every RL step (rollout and minibatch scans) by orders of magnitude.
Walking the jaxpr multiplies each `scan` body by its `length`.

Counted: matmuls (`dot_general`) and convolutions -- the "model FLOPs" that
MFU is quoted in. Elementwise ops, reductions and the optimizer update are
not, which undercounts by a few percent for typical models. The count is of
the *global* computation, independent of how it is sharded.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.extend import core as jcore

from core.distributed import batch_size_of
from core.state import TrainState
from core.task import Task

# Parameter breakdown depth: `params/Block_0` rather than every kernel.
_MODULE_DEPTH = 2


@attrs.frozen(kw_only=True)
class ModuleParams:
    path: str
    count: int


@attrs.frozen(kw_only=True)
class ParamStats:
    count: int
    bytes: int
    # Optimizer state, e.g. Adam's two moments: usually 2x the params.
    optimizer_bytes: int
    modules: tuple[ModuleParams, ...]


@attrs.frozen(kw_only=True)
class FlopStats:
    per_step: float
    # None for a task with no batch dimension (RL's `NoBatch`).
    per_example: float | None
    # A `while_loop` (unknown trip count) was counted once: a lower bound.
    lower_bound: bool


@attrs.frozen(kw_only=True)
class ModelSummary:
    # None when the state is not a `TrainState`.
    params: ParamStats | None
    flops: FlopStats
    # Bytes one train step reads and writes in device memory, per device: XLA's
    # estimate for the compiled step (each fused kernel's inputs and outputs;
    # caches ignored). None if the backend does not report it.
    bytes_per_step: float | None = None


@attrs.frozen(kw_only=True)
class DeviceBandwidth:
    """Peak memory bandwidth of a device, by its `jax.Device.device_kind`."""

    kind: str
    gb_per_sec: float


# Vendor specs. Matched exactly against `device_kind`.
PEAK_BANDWIDTH = (
    DeviceBandwidth(kind="NVIDIA L4", gb_per_sec=300.0),
    DeviceBandwidth(kind="NVIDIA A40", gb_per_sec=696.0),
    DeviceBandwidth(kind="NVIDIA L40", gb_per_sec=864.0),
    DeviceBandwidth(kind="NVIDIA L40S", gb_per_sec=864.0),
    DeviceBandwidth(kind="NVIDIA GeForce RTX 3090", gb_per_sec=936.0),
    DeviceBandwidth(kind="NVIDIA GeForce RTX 4090", gb_per_sec=1008.0),
    DeviceBandwidth(kind="NVIDIA A100 80GB PCIe", gb_per_sec=1935.0),
    DeviceBandwidth(kind="NVIDIA A100-SXM4-80GB", gb_per_sec=2039.0),
    DeviceBandwidth(kind="NVIDIA H100 PCIe", gb_per_sec=2000.0),
    DeviceBandwidth(kind="NVIDIA H100 80GB HBM3", gb_per_sec=3350.0),
)


def peak_bandwidth(kind: str) -> float | None:
    """Peak GB/s of a device kind, or None if not in `PEAK_BANDWIDTH`."""
    return next((d.gb_per_sec for d in PEAK_BANDWIDTH if d.kind == kind), None)


# --- params -------------------------------------------------------------------


def _nbytes(tree: Any) -> int:
    return sum(
        math.prod(np.shape(x)) * np.dtype(x.dtype).itemsize
        for x in jax.tree.leaves(tree)
        if hasattr(x, "dtype") and not jnp.issubdtype(x.dtype, jax.dtypes.prng_key)
    )


def param_stats(params: Any, opt_state: Any = None) -> ParamStats:
    params = nn.unbox(params)
    totals: dict[str, int] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        keys = [jax.tree_util.keystr((k,), simple=True) for k in path[:_MODULE_DEPTH]]
        name = "/".join(keys)
        totals[name] = totals.get(name, 0) + math.prod(np.shape(leaf))
    return ParamStats(
        count=sum(totals.values()),
        bytes=_nbytes(params),
        optimizer_bytes=_nbytes(nn.unbox(opt_state)) if opt_state is not None else 0,
        modules=tuple(ModuleParams(path=p, count=c) for p, c in totals.items()),
    )


# --- flops --------------------------------------------------------------------


@attrs.frozen(kw_only=True)
class _Count:
    flops: float
    lower_bound: bool


def _shape(var: Any) -> tuple[int, ...]:
    """A jaxpr variable's shape. Its `aval` is a ShapedArray for every array
    the counted primitives see; the type checker only knows `AbstractValue`."""
    return tuple(var.aval.shape)


def _dot_flops(eqn: jcore.JaxprEqn) -> float:
    (lhs_contract, _), _ = eqn.params["dimension_numbers"]
    lhs = _shape(eqn.invars[0])
    k = math.prod(lhs[d] for d in lhs_contract)
    return 2.0 * math.prod(_shape(eqn.outvars[0])) * k


def _conv_flops(eqn: jcore.JaxprEqn) -> float:
    rhs = _shape(eqn.invars[1])
    out_feature_dim = eqn.params["dimension_numbers"].rhs_spec[0]
    # Per output element: in-channels-per-group x kernel window, multiply-adds.
    per_output = math.prod(rhs) / rhs[out_feature_dim]
    return 2.0 * math.prod(_shape(eqn.outvars[0])) * per_output


def _sub_jaxprs(value: object) -> Iterable[jcore.Jaxpr]:
    if isinstance(value, jcore.ClosedJaxpr):
        yield value.jaxpr
    elif isinstance(value, jcore.Jaxpr):
        yield value
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _sub_jaxprs(v)


def _count(jaxpr: jcore.Jaxpr) -> _Count:
    flops, lower = 0.0, False
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "dot_general":
            flops += _dot_flops(eqn)
        elif name == "conv_general_dilated":
            flops += _conv_flops(eqn)
        elif name == "cond":
            # One branch runs; count the most expensive.
            branches = [_count(b.jaxpr) for b in eqn.params["branches"]]
            flops += max(b.flops for b in branches)
            lower |= any(b.lower_bound for b in branches)
        else:
            subs = [_count(j) for v in eqn.params.values() for j in _sub_jaxprs(v)]
            inner = sum(s.flops for s in subs)
            lower |= any(s.lower_bound for s in subs)
            if name == "scan":
                inner *= eqn.params["length"]
            elif name == "while" and inner:
                lower = True  # trip count unknown until run time
            flops += inner
    return _Count(flops=flops, lower_bound=lower)


def step_flops(fn: Any, *args: Any) -> _Count:
    """Matmul + conv FLOPs of one call of `fn(*args)`, scans unrolled."""
    return _count(jax.make_jaxpr(fn)(*args).jaxpr)


# --- summary -------------------------------------------------------------------


def step_bytes(fn: Any, *args: Any) -> float | None:
    """XLA's "bytes accessed" for `fn` compiled on `args`, per device.

    Compiles `fn` -- a one-off cost at startup. One step, not the loop's
    scanned chunk: XLA's cost analysis counts a scan body once.
    """
    analysis = jax.jit(fn).lower(*args).compile().cost_analysis()
    if isinstance(analysis, list):  # older jax: one entry per device program
        analysis = analysis[0] if analysis else None
    if not analysis:
        return None
    value = analysis.get("bytes accessed")
    return None if value is None else float(value)


def summarize[StateT, BatchT](
    task: Task[StateT, BatchT, Any, Any], state: StateT, batch: BatchT
) -> ModelSummary:
    """Params of `state`, and FLOPs and memory traffic of one
    `task.train_step` on the global `batch`."""
    count = step_flops(task.train_step, state, batch)
    size = batch_size_of(batch)
    params = (
        param_stats(state.params, state.opt_state)
        if isinstance(state, TrainState)
        else None
    )
    return ModelSummary(
        params=params,
        flops=FlopStats(
            per_step=count.flops,
            per_example=count.flops / size if size else None,
            lower_bound=count.lower_bound,
        ),
        bytes_per_step=step_bytes(task.train_step, state, batch),
    )


def _human(x: float, unit: str = "") -> str:
    for scale, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= scale:
            return f"{x / scale:.2f} {suffix}{unit}"
    return f"{x:.0f} {unit}"


def describe(summary: ModelSummary) -> Sequence[str]:
    """Human-readable lines, for the log at the start of `fit`."""
    lines: list[str] = []
    if summary.params is not None:
        p = summary.params
        lines.append(
            f"params: {p.count:,} ({_human(p.bytes, 'B')}), optimizer state "
            f"{_human(p.optimizer_bytes, 'B')}"
        )
        width = max(len(m.path) for m in p.modules) if p.modules else 0
        for m in sorted(p.modules, key=lambda m: -m.count):
            lines.append(
                f"  {m.path:<{width}}  {m.count:>12,}  {100 * m.count / max(p.count, 1):5.1f}%"
            )
    f = summary.flops
    bound = " (lower bound: a while_loop was counted once)" if f.lower_bound else ""
    per_example = (
        f", {_human(f.per_example, 'FLOP')}/example"
        if f.per_example is not None
        else ""
    )
    lines.append(
        f"train step: {_human(f.per_step, 'FLOP')} (matmul+conv){per_example}{bound}"
    )
    if summary.bytes_per_step is not None:
        lines.append(
            f"train step: {_human(summary.bytes_per_step, 'B')} of device memory "
            "traffic per device (XLA estimate)"
        )
    return lines


def log(summary: ModelSummary) -> None:
    for line in describe(summary):
        logging.info("%s", line)


def device_memory() -> list[tuple[str, float]]:
    """What this process's accelerators actually hold, in GB.

    JAX pre-allocates most of a GPU up front (XLA_PYTHON_CLIENT_MEM_FRACTION,
    75% by default), so `nvidia-smi` shows the pool, not the use. The
    allocator's own counters are the real numbers: `in_use` now, `peak` since
    start (the one to size batches by), and `limit`, the pool. Maxed over local
    devices, which hold the same layout. Empty where the backend keeps no
    stats (CPU).
    """
    stats = [s for d in jax.local_devices() if (s := d.memory_stats()) is not None]
    if not stats:
        return []

    def most(key: str) -> float:
        return max(float(s.get(key, 0)) for s in stats) / 1e9

    return [
        ("device/memory_in_use_gb", most("bytes_in_use")),
        ("device/memory_peak_gb", most("peak_bytes_in_use")),
        ("device/memory_limit_gb", most("bytes_limit")),
    ]
