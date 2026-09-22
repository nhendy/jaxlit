"""Tests for the ragged-to-static array helpers.

Covers numpy and jax alike, plus tracing, because these run both in Grain
workers (numpy) and potentially inside a jitted step (tracers).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from commmon.arrays import pad_to, prefix_mask, zero_invalid, zero_invalid_points
from commmon.fixtures import parameterized_fixture


@parameterized_fixture
def shape() -> dict[str, tuple[int, ...]]:
    """Trailing shapes `pad_to` must preserve untouched."""
    return {
        "scalar_rows": (5,),
        "vector_rows": (5, 2),
        "trajectories": (5, 11, 2),
    }


# --- pad_to -------------------------------------------------------------------


def test_pad_to_grows_the_leading_axis(shape: tuple[int, ...]) -> None:
    out = pad_to(np.ones(shape), 9)
    assert out.shape == (9, *shape[1:])


def test_pad_to_truncates_when_over_budget(shape: tuple[int, ...]) -> None:
    out = pad_to(np.ones(shape), 3)
    assert out.shape == (3, *shape[1:])


def test_pad_to_is_a_noop_at_exact_size(shape: tuple[int, ...]) -> None:
    array = np.ones(shape)
    assert pad_to(array, shape[0]) is array


def test_pad_to_fills_with_zeros(shape: tuple[int, ...]) -> None:
    out = np.asarray(pad_to(np.ones(shape), 9))
    assert out[: shape[0]].all()
    assert not out[shape[0] :].any()


def test_pad_to_preserves_dtype() -> None:
    for dtype in (np.float32, np.int32, bool):
        assert np.asarray(pad_to(np.ones((2,), dtype=dtype), 4)).dtype == dtype


def test_pad_to_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="size"):
        pad_to(np.ones((2, 2)), -1)


def test_pad_to_works_on_jax_arrays(shape: tuple[int, ...]) -> None:
    out = pad_to(jnp.ones(shape), 9)
    assert isinstance(out, jax.Array)
    assert out.shape == (9, *shape[1:])


def test_pad_to_traces(shape: tuple[int, ...]) -> None:
    """The shape branches must be static, not value-dependent."""
    out = jax.jit(lambda x: pad_to(x, 9))(jnp.ones(shape))
    assert out.shape == (9, *shape[1:])


# --- masking ------------------------------------------------------------------


def test_zero_invalid_zeroes_only_masked_entries() -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    valid = np.array([[True, False, True], [False, True, True]])
    out = np.asarray(zero_invalid(array, valid))
    np.testing.assert_array_equal(out[valid], array[valid])
    np.testing.assert_array_equal(out[~valid], 0.0)


def test_zero_invalid_preserves_dtype() -> None:
    out = zero_invalid(np.ones((3,), dtype=np.int32), np.ones((3,), bool))
    assert np.asarray(out).dtype == np.int32


def test_zero_invalid_points_zeroes_whole_points() -> None:
    """A masked point must never end up half-written."""
    array = np.ones((2, 3, 2), dtype=np.float32)
    valid = np.array([[True, False, True], [False, True, True]])
    out = np.asarray(zero_invalid_points(array, valid))
    np.testing.assert_array_equal(out[~valid], 0.0)
    assert (out[valid] == 1.0).all()


def test_zero_invalid_points_rejects_mismatched_mask() -> None:
    with pytest.raises(ValueError):
        zero_invalid_points(np.ones((2, 3, 2)), np.ones((2, 5), bool))


def test_masking_traces() -> None:
    out = jax.jit(zero_invalid_points)(jnp.ones((2, 3, 2)), jnp.ones((2, 3), bool))
    assert out.shape == (2, 3, 2)


# --- prefix_mask --------------------------------------------------------------


def test_prefix_mask_sets_a_prefix() -> None:
    np.testing.assert_array_equal(
        np.asarray(prefix_mask(2, 5)), [True, True, False, False, False]
    )


def test_prefix_mask_endpoints() -> None:
    assert not np.asarray(prefix_mask(0, 4)).any()
    assert np.asarray(prefix_mask(4, 4)).all()


def test_prefix_mask_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="num_true"):
        prefix_mask(5, 4)


def test_prefix_mask_can_build_jax_arrays() -> None:
    """Built by comparison, not slice assignment -- jax arrays reject the latter."""
    out = prefix_mask(2, 5, xnp=jnp)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(np.asarray(out), [True, True, False, False, False])


def test_prefix_mask_agrees_across_backends() -> None:
    np.testing.assert_array_equal(
        np.asarray(prefix_mask(3, 7)), np.asarray(prefix_mask(3, 7, xnp=jnp))
    )


# --- pad/mask together --------------------------------------------------------


def test_pad_then_prefix_mask_agree(shape: tuple[int, ...]) -> None:
    """The two are companions: padding appends, so the mask is a prefix."""
    padded = np.asarray(pad_to(np.ones(shape), 9))
    mask = np.asarray(prefix_mask(shape[0], 9))
    assert padded[mask].all()
    assert not padded[~mask].any()
