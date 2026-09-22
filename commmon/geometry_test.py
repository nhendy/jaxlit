"""Property tests for the geometry primitives.

Mostly invariants rather than hardcoded numbers: rigidity, composition, and
round-trips are what actually break when a sign or a transpose is wrong, and
they hold at every angle rather than the one that happened to be in the fixture.
"""

import numpy as np
import pytest

from commmon.fixtures import parameterized_fixture
from commmon.geometry import (
    arc_length,
    distance_to,
    heading_of,
    resample_polyline,
    rotate,
    rotate_direction,
    rotation_matrix,
    transform_se2,
    wrap_angle,
)


@parameterized_fixture
def angle() -> dict[str, float]:
    """Angles chosen to hit the quadrant boundaries and the wrap seam."""
    return {
        "zero": 0.0,
        "quarter": np.pi / 2,
        "half": np.pi,
        "negative": -np.pi / 3,
        "over_pi": 4.0,
        "tiny": 1e-7,
    }


@pytest.fixture
def points() -> np.ndarray:
    return np.array([[0.0, 0.0], [3.0, 4.0], [-2.0, 1.5], [10.0, -7.0]])


# --- rotation -----------------------------------------------------------------


def test_rotation_matrix_is_orthonormal(angle: float) -> None:
    r = np.asarray(rotation_matrix(angle))
    np.testing.assert_allclose(r @ r.T, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_rotate_by_zero_is_identity(points: np.ndarray) -> None:
    np.testing.assert_allclose(rotate(points, 0.0), points, atol=1e-6)


def test_rotate_preserves_norms(points: np.ndarray, angle: float) -> None:
    """Rigidity: the one property that catches a bad transpose."""
    before = np.linalg.norm(points, axis=-1)
    after = np.linalg.norm(np.asarray(rotate(points, angle)), axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-5)


def test_rotations_compose(points: np.ndarray, angle: float) -> None:
    once = rotate(rotate(points, angle), 0.7)
    twice = rotate(points, angle + 0.7)
    np.testing.assert_allclose(once, twice, atol=1e-5)


def test_rotate_is_counter_clockwise() -> None:
    """Fixes the sign convention: +x rotated by +pi/2 is +y."""
    np.testing.assert_allclose(
        rotate(np.array([[1.0, 0.0]]), np.pi / 2), [[0.0, 1.0]], atol=1e-6
    )


def test_rotate_direction_matches_rotate(points: np.ndarray, angle: float) -> None:
    """Same math, different name -- the distinction is translation, elsewhere."""
    np.testing.assert_allclose(
        rotate_direction(points, angle), rotate(points, angle), atol=1e-6
    )


def test_rotate_accepts_leading_batch_dims(angle: float) -> None:
    """`*n` means the same function serves (a, t, 2) and (2,)."""
    assert np.asarray(rotate(np.zeros((4, 3, 2)), angle)).shape == (4, 3, 2)


# --- angles -------------------------------------------------------------------


def test_wrap_angle_lands_in_range(angle: float) -> None:
    wrapped = np.asarray(
        wrap_angle(np.array([angle, angle + 2 * np.pi, angle - 6 * np.pi]))
    )
    assert np.all(wrapped > -np.pi - 1e-9)
    assert np.all(wrapped <= np.pi + 1e-9)


def test_wrap_angle_is_idempotent(angle: float) -> None:
    once = wrap_angle(np.array([angle]))
    np.testing.assert_allclose(wrap_angle(once), once, atol=1e-9)


def test_wrap_angle_preserves_direction(angle: float) -> None:
    """Wrapping must not change what the angle means, only its representation."""
    wrapped = float(np.asarray(wrap_angle(np.array([angle])))[0])
    np.testing.assert_allclose(
        [np.cos(wrapped), np.sin(wrapped)],
        [np.cos(angle), np.sin(angle)],
        atol=1e-6,
    )


def test_heading_of_inverts_rotation(angle: float) -> None:
    unit = np.asarray(rotate(np.array([[1.0, 0.0]]), angle))
    np.testing.assert_allclose(
        heading_of(unit), wrap_angle(np.array([angle])), atol=1e-5
    )


# --- SE(2) --------------------------------------------------------------------


def test_transform_se2_sends_translation_to_origin(
    points: np.ndarray, angle: float
) -> None:
    out = np.asarray(transform_se2(points, points[1], angle))
    np.testing.assert_allclose(out[1], [0.0, 0.0], atol=1e-5)


def test_transform_se2_is_rigid(points: np.ndarray, angle: float) -> None:
    out = np.asarray(transform_se2(points, np.array([1.0, -2.0]), angle))
    before = np.linalg.norm(points[:, None] - points[None], axis=-1)
    after = np.linalg.norm(out[:, None] - out[None], axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-5)


def test_transform_se2_round_trips(points: np.ndarray, angle: float) -> None:
    translation = np.array([3.0, -1.0])
    forward = np.asarray(transform_se2(points, translation, angle))
    back = np.asarray(rotate(forward, -angle)) + translation
    np.testing.assert_allclose(back, points, atol=1e-4)


def test_transform_se2_translates_before_rotating() -> None:
    """Order matters; the reverse would give [0, -1] here."""
    out = transform_se2(np.array([[1.0, 0.0]]), np.array([1.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(out, [[1.0, 0.0]], atol=1e-6)


# --- polylines ----------------------------------------------------------------


def test_arc_length_starts_at_zero_and_increases() -> None:
    lengths = np.asarray(arc_length(np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]])))
    assert lengths[0] == 0.0
    np.testing.assert_allclose(lengths, [0.0, 5.0, 11.0], atol=1e-6)
    assert np.all(np.diff(lengths) >= 0)


def test_resample_hits_the_requested_length() -> None:
    assert np.asarray(
        resample_polyline(np.array([[0.0, 0.0], [10.0, 0.0]]), 20)
    ).shape == (20, 2)


def test_resample_preserves_endpoints() -> None:
    line = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]])
    out = np.asarray(resample_polyline(line, 7))
    np.testing.assert_allclose(out[0], line[0], atol=1e-5)
    np.testing.assert_allclose(out[-1], line[-1], atol=1e-5)


def test_resample_is_arc_length_uniform() -> None:
    """Uniform spacing is what makes a resampled polyline usable as a feature."""
    line = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    steps = np.linalg.norm(
        np.diff(np.asarray(resample_polyline(line, 11)), axis=0), axis=-1
    )
    np.testing.assert_allclose(steps, steps[0], rtol=1e-4)


def test_resample_handles_single_point() -> None:
    out = np.asarray(resample_polyline(np.array([[2.0, 3.0]]), 5))
    np.testing.assert_allclose(out, np.tile([2.0, 3.0], (5, 1)))


def test_resample_handles_zero_length_polyline() -> None:
    """Repeated identical vertices appear in real maps and would divide by zero."""
    out = np.asarray(resample_polyline(np.array([[1.0, 1.0], [1.0, 1.0]]), 4))
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


def test_resample_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="num_points"):
        resample_polyline(np.array([[0.0, 0.0], [1.0, 0.0]]), 0)


def test_resample_commutes_with_rotation(angle: float) -> None:
    """Resampling is arc-length based, so it cannot depend on orientation."""
    line = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 2.0]])
    a = np.asarray(rotate(resample_polyline(line, 9), angle))
    b = np.asarray(resample_polyline(np.asarray(rotate(line, angle)), 9))
    np.testing.assert_allclose(a, b, atol=1e-4)


# --- distances ----------------------------------------------------------------


def test_distance_to(points: np.ndarray) -> None:
    np.testing.assert_allclose(
        distance_to(points, np.zeros(2)), np.linalg.norm(points, axis=-1), atol=1e-6
    )


def test_distance_to_self_is_zero(points: np.ndarray) -> None:
    assert float(np.asarray(distance_to(points, points[2]))[2]) == pytest.approx(0.0)


def test_distance_is_rotation_invariant(points: np.ndarray, angle: float) -> None:
    rotated = np.asarray(rotate(points, angle))
    np.testing.assert_allclose(
        distance_to(rotated, rotated[0]), distance_to(points, points[0]), atol=1e-5
    )


# --- shape checking -----------------------------------------------------------


def test_rotate_rejects_non_2d_points() -> None:
    with pytest.raises(ValueError, match="dim"):
        rotate(np.zeros((3, 3)), 0.1)


def test_distance_to_rejects_bad_reference() -> None:
    with pytest.raises(ValueError, match="reference"):
        distance_to(np.zeros((3, 2)), np.zeros(3))


def test_transform_se2_rejects_bad_translation() -> None:
    with pytest.raises(ValueError, match="translation"):
        transform_se2(np.zeros((3, 2)), np.zeros(3), 0.1)


# --- numpy / jax parity -------------------------------------------------------
#
# The whole point of dispatching through etils.enp is that these are reusable
# inside a jitted training step, not just in a Grain worker. Parity and
# traceability are therefore part of the contract, not a nice-to-have.


@pytest.fixture
def jnp_module():
    import jax.numpy as jnp

    return jnp


def test_rotate_matches_between_numpy_and_jax(
    points: np.ndarray, angle: float, jnp_module
) -> None:
    np.testing.assert_allclose(
        np.asarray(rotate(jnp_module.asarray(points), jnp_module.float32(angle))),
        np.asarray(rotate(points, angle)),
        atol=1e-4,
    )


def test_resample_matches_between_numpy_and_jax(jnp_module) -> None:
    line = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 2.0]])
    np.testing.assert_allclose(
        np.asarray(resample_polyline(jnp_module.asarray(line), 9)),
        np.asarray(resample_polyline(line, 9)),
        atol=1e-4,
    )


def test_geometry_traces_under_jit(jnp_module) -> None:
    """Value branches or np.asarray on an input would raise here."""
    import jax

    @jax.jit
    def body(points, angle):
        moved = transform_se2(points, jnp_module.zeros(2), angle)
        return distance_to(moved[:, 0], jnp_module.zeros(2)), heading_of(moved[:, 0])

    dist, head = body(jnp_module.ones((3, 4, 2)), jnp_module.float32(0.3))
    assert dist.shape == (3,) and head.shape == (3,)


def test_degenerate_polyline_traces_under_jit(jnp_module) -> None:
    """The zero-length guard is a clamp, not a branch -- so it must trace."""
    import jax

    out = jax.jit(lambda line: resample_polyline(line, 4))(
        jnp_module.array([[1.0, 1.0], [1.0, 1.0]])
    )
    assert out.shape == (4, 2)
    assert bool(np.isfinite(out).all())


def test_jax_output_stays_a_jax_array(points: np.ndarray, jnp_module) -> None:
    """Dispatch must not silently round-trip through numpy."""
    import jax

    out = rotate(jnp_module.asarray(points), jnp_module.float32(0.4))
    assert isinstance(out, jax.Array)


# --- array angles -------------------------------------------------------------
#
# `angle` is an array, not a float. That is what lets eincheck check it, lets the
# module dispatch numpy/jax on it, and makes a batch of per-agent angles as
# ordinary as a single one.


def test_rotation_matrix_batches_over_angles() -> None:
    for shape in ((), (3,), (4, 3)):
        out = np.asarray(rotation_matrix(np.zeros(shape)))
        assert out.shape == (*shape, 2, 2)


def test_rotation_matrix_batch_matches_elementwise() -> None:
    angles = np.array([0.0, 0.7, -2.1])
    batched = np.asarray(rotation_matrix(angles))
    for i, a in enumerate(angles):
        np.testing.assert_allclose(
            batched[i], np.asarray(rotation_matrix(np.float64(a))), atol=1e-6
        )


def test_rotate_agrees_with_the_matrix_form(points: np.ndarray, angle: float) -> None:
    """Component math and `points @ R.T` must not drift apart."""
    a = np.float64(angle)
    np.testing.assert_allclose(
        rotate(points, a), points @ np.asarray(rotation_matrix(a)).T, atol=1e-6
    )


def test_rotate_broadcasts_a_per_row_angle() -> None:
    """The capability an array angle buys: each agent rotated by its own."""
    points = np.tile(np.array([[1.0, 0.0]]), (3, 4, 1))
    angles = np.array([0.0, np.pi / 2, np.pi])[:, None]
    out = np.asarray(rotate(points, angles))
    assert out.shape == (3, 4, 2)
    np.testing.assert_allclose(
        out[:, 0], [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], atol=1e-6
    )


def test_per_row_angle_matches_rotating_each_row_alone() -> None:
    rng = np.random.default_rng(0)
    points = rng.standard_normal((5, 6, 2))
    angles = rng.uniform(-np.pi, np.pi, size=(5, 1))
    batched = np.asarray(rotate(points, angles))
    for i in range(5):
        np.testing.assert_allclose(
            batched[i], np.asarray(rotate(points[i], angles[i, 0])), atol=1e-6
        )


def test_zero_d_angle_rotates_everything_uniformly() -> None:
    points = np.ones((3, 4, 2))
    np.testing.assert_allclose(
        rotate(points, np.float64(0.4)), rotate(points, np.array(0.4)), atol=1e-9
    )


def test_transform_se2_takes_a_per_row_angle() -> None:
    points = np.tile(np.array([[2.0, 0.0]]), (2, 3, 1))
    angles = np.array([0.0, np.pi])[:, None]
    out = np.asarray(transform_se2(points, np.array([1.0, 0.0]), angles))
    np.testing.assert_allclose(out[:, 0], [[1.0, 0.0], [-1.0, 0.0]], atol=1e-6)


def test_angle_shape_is_independent_of_points_shape() -> None:
    """`*m` is a separate variadic from `*n`; they need not match.

    The rule is that `angle` broadcasts against `points[..., 0]` -- the point
    shape *minus* the coordinate axis. So `(7, 2)` points take a `()` or `(7,)`
    angle, and `(a, t, 2)` points take `(a, 1)` for one angle per agent.
    """
    assert np.asarray(rotate(np.zeros((7, 2)), np.zeros(()))).shape == (7, 2)
    assert np.asarray(rotate(np.zeros((7, 2)), np.zeros(7))).shape == (7, 2)
    assert np.asarray(rotate(np.zeros((5, 6, 2)), np.zeros((5, 1)))).shape == (5, 6, 2)


def test_a_misaligned_angle_is_rejected() -> None:
    """`(7, 1)` against `(7, 2)` points broadcasts to `(7, 7, 2)` -- a real
    broadcast, but never the intended one. The output spec is what catches it."""
    with pytest.raises(ValueError, match="output0"):
        rotate(np.zeros((7, 2)), np.zeros((7, 1)))


def test_batched_angles_trace_under_jit(jnp_module) -> None:
    import jax

    out = jax.jit(rotate)(jnp_module.ones((3, 4, 2)), jnp_module.zeros((3, 1)))
    assert out.shape == (3, 4, 2)
