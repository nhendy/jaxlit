"""Array shaping and masking helpers, shape-checked with eincheck.

The ragged-to-static operations every JAX preprocessing pipeline needs, in one
place so the padding and masking conventions cannot drift between call sites.

Like `commmon.geometry`, these are numpy/jax agnostic -- the module is resolved
per call via `etils.enp.lazy.get_xnp`, so they work unchanged inside `jax.jit`.
No input is ever passed through `np.asarray` (it would concretize a tracer) and
no branch tests a *value*; the shape branches below are static under trace.
"""

import eincheck
import jaxtyping as jt
import numpy as np

from commmon.geometry import xnp_for


@eincheck.check_func("a *rest, -> b *rest")
def pad_to(array: jt.ArrayLike, size: int) -> jt.ArrayLike:
    """Zero-pad (or truncate) the leading axis to exactly `size`.

    The trailing shape is preserved, which is what makes this safe to apply to
    `(a, t, 2)` trajectories and `(a,)` labels alike. Both branches below test
    `array.shape`, which jax knows statically, so this traces.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")

    if array.shape[0] == size:
        return array
    if array.shape[0] > size:
        return array[:size]

    xnp = xnp_for(array)
    widths = [(0, size - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
    return xnp.pad(array, widths)


@eincheck.check_func("*n, *n -> *n")
def zero_invalid(array: jt.ArrayLike, valid: jt.ArrayLike) -> jt.ArrayLike:
    """Zero every entry where `valid` is False, preserving dtype.

    Invalid entries are masked everywhere downstream, so their contents are
    arbitrary -- but leaving stale values in them means a bug that ignores the
    mask produces plausible numbers instead of obviously wrong ones. Zeroing
    makes that failure loud.

    For trailing feature axes (e.g. `(a, t)` validity over `(a, t, 2)` points)
    use `zero_invalid_points`, which broadcasts the mask for you.
    """
    xnp = xnp_for(array, valid)
    return xnp.where(valid, array, xnp.zeros((), dtype=array.dtype))


@eincheck.check_func("*n k, *n -> *n k")
def zero_invalid_points(array: jt.ArrayLike, valid: jt.ArrayLike) -> jt.ArrayLike:
    """`zero_invalid` for arrays with one trailing feature axis.

    The mask is per-point; the feature axis is zeroed as a unit, so a masked
    point never ends up half-written.
    """
    xnp = xnp_for(array, valid)
    return xnp.where(valid[..., None], array, xnp.zeros((), dtype=array.dtype))


@eincheck.check_func("-> a")
def prefix_mask(num_true: int, size: int, *, xnp=np) -> jt.ArrayLike:
    """Boolean mask with the first `num_true` entries set.

    The companion to `pad_to`: padding appends at the end, so a valid-row mask is
    always a prefix. Built by comparison rather than slice assignment, which jax
    arrays do not support.
    """
    if not 0 <= num_true <= size:
        raise ValueError(f"num_true {num_true} outside [0, {size}]")
    return xnp.arange(size) < num_true
