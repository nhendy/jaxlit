"""2D geometry primitives, shape-checked with eincheck, numpy/jax agnostic.

Every function here is general: it takes plain arrays, knows nothing about
`Scenario` or any dataset, and accepts arbitrary leading dimensions via the
named variadic `*n`. That is deliberate -- preprocessing code should compose
these rather than re-deriving rotation matrices and arc-length interpolation at
each call site, where the conventions inevitably drift apart.

**These run under `jax.jit`.** The numpy module is resolved per call with
`etils.enp.lazy.get_xnp`, so the same function serves a Grain worker holding
numpy arrays and a jitted training step holding tracers. Two rules follow, and
both are easy to break by accident:

* never call `np.asarray` on an input -- it would concretize a tracer;
* never branch on a *value* -- only on shapes, which are static under trace.

eincheck itself traces fine: its checks are on shapes, which jax knows
statically, so `@check_func` survives `jit` unchanged.

Conventions, fixed once here so callers cannot disagree:

* Points are `(x, y)` in the trailing axis.
* Angles are radians, counter-clockwise positive.
* A rotation matrix is applied as `points @ R.T`, so the same `R` works for a
  single point and for any stack of them.
* Headings are kept wrapped to (-pi, pi]; unwrapped angles accumulate error
  across composed transforms and break any loss that differences them.
"""

import eincheck
import jaxtyping as jt
import numpy as np
from etils import enp

_EPS = 1e-12


def xnp_for(*arrays: jt.ArrayLike):
    """The array module backing `arrays` -- `numpy` or `jax.numpy`.

    Falls back to numpy for python scalars, so a caller mixing a traced array
    with a float literal still dispatches on the array.
    """
    for array in arrays:
        module = enp.lazy.get_xnp(array, strict=False)
        if module is not np:
            return module
    return np


@eincheck.check_func("*n -> *n 2 2")
def rotation_matrix(angle: jt.ArrayLike) -> jt.ArrayLike:
    """Counter-clockwise rotation matrices, for use as `points @ R.T`.

    `angle` is an array, not a python float: that is what lets eincheck check it,
    lets the module dispatch on it, and makes a *batch* of angles -- one per
    agent, say -- as ordinary as a single one. A 0-d angle yields `(2, 2)`;
    shape `(a,)` yields `(a, 2, 2)`.
    """
    xnp = xnp_for(angle)
    cos, sin = xnp.cos(angle), xnp.sin(angle)
    return xnp.stack(
        [xnp.stack([cos, -sin], axis=-1), xnp.stack([sin, cos], axis=-1)], axis=-2
    )


@eincheck.check_func("*n -> *n")
def wrap_angle(angle: jt.ArrayLike) -> jt.ArrayLike:
    """Wrap radians to (-pi, pi].

    Composed rotations otherwise drift outside the range, and any downstream
    angular difference silently picks the long way round.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@eincheck.check_func("*n 2, *m -> *n 2")
def rotate(points: jt.ArrayLike, angle: jt.ArrayLike) -> jt.ArrayLike:
    """Rotate points about the origin.

    `angle` carries its own shape (`*m`), independent of the points' (`*n`).
    The rule is that it broadcasts against `points[..., 0]` -- the point shape
    *minus* the coordinate axis. So `(a, t, 2)` points take a 0-d angle to rotate
    everything alike, or `(a, 1)` to give each agent its own.

    A misaligned angle -- `(a, 1)` against `(a, 2)` points -- still broadcasts,
    to `(a, a, 2)`, which is never what was meant. The output spec catches it.

    Implemented component-wise rather than as `points @ rotation_matrix(angle).T`
    precisely so that broadcasting works: a batched matrix would need the point
    and angle axes aligned by hand.
    """
    xnp = xnp_for(points, angle)
    cos, sin = xnp.cos(angle), xnp.sin(angle)
    x, y = points[..., 0], points[..., 1]
    return xnp.stack([x * cos - y * sin, x * sin + y * cos], axis=-1)


@eincheck.check_func("*n 2, 2, *m -> *n 2")
def transform_se2(
    points: jt.ArrayLike, translation: jt.ArrayLike, angle: jt.ArrayLike
) -> jt.ArrayLike:
    """Translate by `-translation`, then rotate by `angle`.

    The rigid-body change of frame used to put a scenario in some agent's
    reference frame. Translate-then-rotate, not the reverse: the result is the
    old frame expressed in the new one.
    """
    return rotate(points - translation, angle)


@eincheck.check_func("*n 2, *m -> *n 2")
def rotate_direction(vectors: jt.ArrayLike, angle: jt.ArrayLike) -> jt.ArrayLike:
    """Rotate direction vectors (velocities, normals).

    Identical to `rotate` but named for the call site: directions must never be
    translated, and mixing the two up is a bug that produces plausible-looking
    output.
    """
    return rotate(vectors, angle)


@eincheck.check_func("p 2 -> p")
def arc_length(points: jt.ArrayLike) -> jt.ArrayLike:
    """Cumulative distance along a polyline, starting at 0."""
    xnp = xnp_for(points)
    segment = xnp.linalg.norm(xnp.diff(points, axis=0), axis=-1)
    return xnp.concatenate([xnp.zeros((1,), dtype=segment.dtype), xnp.cumsum(segment)])


@eincheck.check_func("p 2, -> q 2")
def resample_polyline(points: jt.ArrayLike, num_points: int) -> jt.ArrayLike:
    """Resample a polyline to exactly `num_points`, uniform in arc length.

    Raw map polylines vary wildly in vertex count while a fixed-shape schema
    wants one number. Uniform arc-length spacing preserves the geometry and
    leaves every output point meaningful, so there is no ragged padding inside a
    polyline.

    Degenerate polylines are real -- maps contain single-vertex and
    repeated-vertex entries. The zero-length case is handled by clamping the
    total rather than branching on it, because a value branch would not trace;
    interpolating against a clamped total returns the first vertex repeated,
    which is the same answer.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")

    xnp = xnp_for(points)
    if points.shape[0] == 1:  # a shape, so this branch is static under jit
        return xnp.repeat(points, num_points, axis=0)

    distance = arc_length(points)
    total = xnp.maximum(distance[-1], _EPS)
    wanted = xnp.linspace(0.0, total, num_points)
    return xnp.stack(
        [xnp.interp(wanted, distance, points[:, axis]) for axis in range(2)], axis=-1
    )


@eincheck.check_func("a 2, 2 -> a")
def distance_to(points: jt.ArrayLike, reference: jt.ArrayLike) -> jt.ArrayLike:
    """Euclidean distance from each point to a single reference point."""
    xnp = xnp_for(points, reference)
    return xnp.linalg.norm(points - reference, axis=-1)


@eincheck.check_func("*n 2 -> *n")
def heading_of(vectors: jt.ArrayLike) -> jt.ArrayLike:
    """Heading implied by a direction vector, wrapped to (-pi, pi]."""
    xnp = xnp_for(vectors)
    return wrap_angle(xnp.arctan2(vectors[..., 1], vectors[..., 0]))
