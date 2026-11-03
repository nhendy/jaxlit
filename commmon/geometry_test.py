"""Property tests for the geometry primitives.

Mostly invariants rather than hardcoded numbers: rigidity, composition, and
round-trips are what actually break when a sign or a transpose is wrong, and
they hold at every angle rather than the one that happened to be in the fixture.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from commmon.fixtures import parameterized_fixture
from commmon.geometry import (
    arc_length,
    box_corners,
    chunk_polyline,
    distance_to,
    heading_of,
    resample_polyline,
    rotate,
    rotate_direction,
    rotation_matrix,
    transform_se2,
    union_outline,
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


# --- oriented boxes -----------------------------------------------------------


@pytest.fixture
def box() -> dict[str, np.ndarray]:
    """A 4x2 m box at the origin, axis aligned -- easy to reason about by hand."""
    return {
        "center": np.zeros(2),
        "extent": np.array([4.0, 2.0]),
    }


def test_box_corners_at_zero_heading(box: dict[str, np.ndarray]) -> None:
    """Pins both the corner order and which half-extent is which."""
    out = np.asarray(box_corners(box["center"], np.float64(0.0), box["extent"]))
    np.testing.assert_allclose(
        out, [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]], atol=1e-6
    )


def test_box_length_lies_along_the_heading(
    box: dict[str, np.ndarray], angle: float
) -> None:
    """The failure this catches: length and width swapped."""
    out = np.asarray(box_corners(box["center"], np.float64(angle), box["extent"]))
    forward = np.asarray(rotate(np.array([1.0, 0.0]), np.float64(angle)))
    along = (out[1] + out[2]) / 2.0 - (out[0] + out[3]) / 2.0
    np.testing.assert_allclose(along, 4.0 * forward, atol=1e-5)


def test_box_edges_keep_their_lengths(box: dict[str, np.ndarray], angle: float) -> None:
    """Rigidity: rotation must not stretch the box."""
    out = np.asarray(box_corners(box["center"], np.float64(angle), box["extent"]))
    edges = np.linalg.norm(np.diff(np.concatenate([out, out[:1]]), axis=0), axis=-1)
    np.testing.assert_allclose(edges, [4.0, 2.0, 4.0, 2.0], atol=1e-5)


def test_box_corners_stay_centred(box: dict[str, np.ndarray], angle: float) -> None:
    center = np.array([3.0, -7.0])
    out = np.asarray(box_corners(center, np.float64(angle), box["extent"]))
    np.testing.assert_allclose(out.mean(axis=0), center, atol=1e-5)


def test_box_corners_are_counter_clockwise(
    box: dict[str, np.ndarray], angle: float
) -> None:
    """Shoelace area is positive only for the counter-clockwise winding."""
    out = np.asarray(box_corners(box["center"], np.float64(angle), box["extent"]))
    shifted = np.roll(out, -1, axis=0)
    area = 0.5 * np.sum(out[:, 0] * shifted[:, 1] - shifted[:, 0] * out[:, 1])
    assert area == pytest.approx(8.0, abs=1e-4)


def test_box_corners_rotate_with_the_frame(
    box: dict[str, np.ndarray], angle: float
) -> None:
    """Building in one frame then rotating must match building in the other."""
    turned = np.asarray(box_corners(box["center"], np.float64(angle), box["extent"]))
    direct = np.asarray(
        rotate(box_corners(box["center"], np.float64(0.0), box["extent"]), angle)
    )
    np.testing.assert_allclose(turned, direct, atol=1e-5)


def test_box_corners_batch_over_agents() -> None:
    out = np.asarray(box_corners(np.zeros((5, 2)), np.zeros(5), np.ones((5, 2))))
    assert out.shape == (5, 4, 2)


def test_one_extent_serves_a_whole_trajectory() -> None:
    """The broadcast that matters: extent is per agent, pose is per step."""
    out = np.asarray(
        box_corners(np.zeros((5, 6, 2)), np.zeros((5, 6)), np.ones((5, 1, 2)))
    )
    assert out.shape == (5, 6, 4, 2)


def test_batched_box_matches_one_at_a_time() -> None:
    rng = np.random.default_rng(0)
    center = rng.standard_normal((4, 2))
    heading = rng.uniform(-np.pi, np.pi, size=4)
    extent = rng.uniform(1.0, 5.0, size=(4, 2))
    batched = np.asarray(box_corners(center, heading, extent))
    for i in range(4):
        np.testing.assert_allclose(
            batched[i],
            np.asarray(box_corners(center[i], heading[i], extent[i])),
            atol=1e-6,
        )


def test_box_corners_reject_a_misaligned_extent() -> None:
    """A per-agent `(a, 2)` extent against `(a, t, 2)` centres needs `(a, 1, 2)`.

    Without the explicit broadcast it would silently align `a` against `t`.
    """
    with pytest.raises(ValueError, match="broadcast"):
        box_corners(np.zeros((4, 3, 2)), np.zeros((4, 3)), np.ones((4, 2)))


def test_a_shared_extent_serves_every_box() -> None:
    out = np.asarray(box_corners(np.zeros((5, 2)), np.zeros(5), np.array([4.0, 2.0])))
    assert out.shape == (5, 4, 2)
    np.testing.assert_allclose(out[0], out[-1], atol=1e-6)


def test_box_corners_trace_under_jit(jnp_module) -> None:

    out = jax.jit(box_corners)(
        jnp_module.zeros((3, 2)), jnp_module.zeros(3), jnp_module.ones((3, 2))
    )
    assert out.shape == (3, 4, 2)


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


# --- chunking -----------------------------------------------------------------


@pytest.fixture
def long_line() -> np.ndarray:
    """A 400 m straight -- far longer than one 20-point chunk can resolve."""
    return np.array([[0.0, 0.0], [400.0, 0.0]])


def test_chunking_respects_the_spacing_cap(long_line: np.ndarray) -> None:
    out = np.asarray(chunk_polyline(long_line, 20, 2.0))
    step = np.linalg.norm(np.diff(out, axis=1), axis=-1)
    assert step.max() <= 2.0 + 1e-6, "a chunk exceeded max_spacing"


def test_chunk_count_is_the_minimum_that_fits(long_line: np.ndarray) -> None:
    """One chunk spans `max_spacing * (num_points - 1)` = 38 m, so 400 m needs 11."""
    assert np.asarray(chunk_polyline(long_line, 20, 2.0)).shape == (11, 20, 2)


def test_chunks_share_endpoints(long_line: np.ndarray) -> None:
    """The seam invariant: a gap here would read as a break in the road."""
    out = np.asarray(chunk_polyline(long_line, 20, 2.0))
    np.testing.assert_allclose(out[:-1, -1], out[1:, 0], atol=1e-6)


def test_chunks_reassemble_into_the_original(angle: float) -> None:
    line = np.asarray(
        rotate(np.array([[0.0, 0.0], [120.0, 0.0], [120.0, 90.0]]), angle)
    )
    out = np.asarray(chunk_polyline(line, 20, 2.0))
    # Dropping each chunk's shared first point leaves one unbroken polyline.
    joined = np.concatenate([out[0], *(chunk[1:] for chunk in out[1:])])
    np.testing.assert_allclose(joined[0], line[0], atol=1e-4)
    np.testing.assert_allclose(joined[-1], line[-1], atol=1e-4)
    # Chording a corner can only shorten the path, and by at most about one
    # spacing per corner -- this line has one.
    total = float(np.asarray(arc_length(line))[-1])
    joined_length = np.linalg.norm(np.diff(joined, axis=0), axis=-1).sum()
    assert total - 2.0 <= joined_length <= total + 1e-6


def test_a_short_polyline_is_one_chunk() -> None:
    """Below the cap nothing is split, so lanes stay one polyline each."""
    line = np.array([[0.0, 0.0], [20.0, 0.0]])
    assert np.asarray(chunk_polyline(line, 20, 2.0)).shape == (1, 20, 2)


def test_infinite_spacing_is_plain_resampling() -> None:
    """The opt-out a dataset uses when its polyline budget binds."""
    line = np.array([[0.0, 0.0], [500.0, 0.0], [500.0, 250.0]])
    out = np.asarray(chunk_polyline(line, 20, float("inf")))
    assert out.shape == (1, 20, 2)
    np.testing.assert_allclose(
        out[0], np.asarray(resample_polyline(line, 20)), atol=1e-6
    )


def test_chunking_beats_resampling_on_a_long_ring() -> None:
    """The bug this exists for: a fixed budget chords straight across the road.

    Shaped like a real drivable area -- a long, narrow ring around a street --
    rather than a circle. A circle is the *kindest* case for uniform
    resampling; an elongated ring is where 20 points start cutting corners off
    entirely.
    """
    corners = np.array(
        [[0.0, 0.0], [200.0, 0.0], [200.0, 12.0], [0.0, 12.0], [0.0, 0.0]]
    )
    ring = np.concatenate(
        [
            np.linspace(a, b, max(2, int(np.linalg.norm(b - a))), endpoint=False)
            for a, b in zip(corners[:-1], corners[1:])
        ]
        + [corners[-1:]]
    )

    def deviation(polyline: np.ndarray) -> float:
        a, b = polyline[:-1], polyline[1:]
        d = b - a
        # Chunk seams repeat a point, so some segments have zero length.
        length = np.maximum((d * d).sum(-1), 1e-12)
        u = np.clip(((ring[:, None] - a) * d).sum(-1) / length, 0.0, 1.0)
        near = a + u[..., None] * d
        return float(np.linalg.norm(ring[:, None] - near, axis=-1).min(axis=1).max())

    flat = deviation(np.asarray(resample_polyline(ring, 20)))
    chunked = deviation(np.asarray(chunk_polyline(ring, 20, 2.0)).reshape(-1, 2))

    assert flat > 5.0, "fixture is not long enough to show the problem"
    # What is left is the corner cut, not the spacing: a 2 m chord across a
    # right angle loses at most ~1 m however many chunks there are. Real AV2
    # drivable areas bottom out in the same place (p99 0.70 m).
    assert chunked < 1.0
    assert chunked < flat / 10.0


def test_chunking_handles_a_single_point() -> None:
    """Stop signs are one vertex; the degenerate answer is that point repeated."""
    out = np.asarray(chunk_polyline(np.array([[3.0, 4.0]]), 20, 2.0))
    assert out.shape == (1, 20, 2)
    np.testing.assert_allclose(out[0], np.tile([3.0, 4.0], (20, 1)), atol=1e-6)


def test_chunking_handles_a_zero_length_polyline() -> None:
    out = np.asarray(chunk_polyline(np.array([[1.0, 1.0], [1.0, 1.0]]), 20, 2.0))
    assert out.shape == (1, 20, 2)
    assert np.isfinite(out).all()


def test_chunking_rejects_a_degenerate_chunk_size() -> None:
    with pytest.raises(ValueError, match="num_points"):
        chunk_polyline(np.array([[0.0, 0.0], [1.0, 0.0]]), 1, 2.0)


def test_chunking_rejects_non_positive_spacing() -> None:
    with pytest.raises(ValueError, match="max_spacing"):
        chunk_polyline(np.array([[0.0, 0.0], [1.0, 0.0]]), 20, 0.0)


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

    @jax.jit
    def body(points, angle):
        moved = transform_se2(points, jnp_module.zeros(2), angle)
        return distance_to(moved[:, 0], jnp_module.zeros(2)), heading_of(moved[:, 0])

    dist, head = body(jnp_module.ones((3, 4, 2)), jnp_module.float32(0.3))
    assert dist.shape == (3,) and head.shape == (3,)


def test_degenerate_polyline_traces_under_jit(jnp_module) -> None:
    """The zero-length guard is a clamp, not a branch -- so it must trace."""

    out = jax.jit(lambda line: resample_polyline(line, 4))(
        jnp_module.array([[1.0, 1.0], [1.0, 1.0]])
    )
    assert out.shape == (4, 2)
    assert bool(np.isfinite(out).all())


def test_jax_output_stays_a_jax_array(points: np.ndarray, jnp_module) -> None:
    """Dispatch must not silently round-trip through numpy."""

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

    out = jax.jit(rotate)(jnp_module.ones((3, 4, 2)), jnp_module.zeros((3, 1)))
    assert out.shape == (3, 4, 2)


def _square(x0: float, y0: float) -> np.ndarray:
    return np.array([[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1]])


def _length(polylines: list[np.ndarray]) -> float:
    return sum(
        float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()) for p in polylines
    )


def test_union_outline_cancels_shared_sides() -> None:
    # Two unit squares side by side: the seam at x=1 is interior.
    out = union_outline([_square(0, 0), _square(1, 0)])
    assert len(out) == 1
    assert _length(out) == pytest.approx(6.0)  # 2x1 rectangle, no seam
    np.testing.assert_allclose(out[0][0], out[0][-1])  # comes back closed
    assert not any(np.allclose(v, [1.0, 0.5]) for v in out[0])


def test_union_outline_keeps_separate_polygons_separate() -> None:
    out = union_outline([_square(0, 0), _square(5, 0)])
    assert len(out) == 2
    assert _length(out) == pytest.approx(8.0)


def test_union_outline_accepts_closed_rings_and_float_noise() -> None:
    a = np.vstack([_square(0, 0), [[0.0, 0.0]]])  # explicitly closed
    b = _square(1, 0) + 1e-6  # same seam, off by float noise
    assert _length(union_outline([a, b])) == pytest.approx(6.0, abs=1e-4)


def test_union_outline_of_a_grid_is_its_perimeter() -> None:
    tiles = [_square(x, y) for x in range(3) for y in range(2)]
    out = union_outline(tiles)
    assert len(out) == 1
    assert _length(out) == pytest.approx(10.0)
