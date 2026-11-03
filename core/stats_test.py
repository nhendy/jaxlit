"""FLOP counting from the jaxpr: matmuls, convs, scans, conds; parameter counts."""

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from core import stats
from core.stats import param_stats, step_flops
from core.types import Variables


def test_matmul() -> None:
    a, b = jnp.ones((8, 32)), jnp.ones((32, 16))
    assert step_flops(jnp.matmul, a, b).flops == 2 * 8 * 32 * 16


def test_scan_multiplies_by_length() -> None:
    """XLA's cost analysis counts a scan body once; this must not."""

    def body(w: jax.Array, x: jax.Array) -> tuple[jax.Array, None]:
        return x @ w, None

    w, xs = jnp.ones((16, 16)), jnp.ones((10, 16, 16))
    count = step_flops(lambda w, xs: jax.lax.scan(body, w, xs)[0], w, xs)
    assert count.flops == 10 * 2 * 16**3
    assert not count.lower_bound


def test_nested_scan_and_cond() -> None:
    def inner(c: jax.Array, _: None) -> tuple[jax.Array, None]:
        return c @ c, None

    def outer(c: jax.Array, _: None) -> tuple[jax.Array, None]:
        return jax.lax.scan(inner, c, length=3)[0], None

    def f(c: jax.Array, p: jax.Array) -> jax.Array:
        c = jax.lax.scan(outer, c, length=4)[0]
        # cond: the costlier branch counts.
        return jax.lax.cond(p, lambda c: c @ c @ c, lambda c: c, c)

    count = step_flops(f, jnp.ones((8, 8)), jnp.asarray(True))
    assert count.flops == (4 * 3 + 2) * 2 * 8**3


def test_while_is_a_lower_bound() -> None:
    def f(c: jax.Array) -> jax.Array:
        return jax.lax.while_loop(lambda c: c.sum() < 1e9, lambda c: c @ c, c)

    count = step_flops(f, jnp.ones((4, 4)))
    assert count.lower_bound
    assert count.flops == 2 * 4**3


def test_conv() -> None:
    x = jnp.ones((2, 16, 16, 3))
    conv = nn.Conv(features=8, kernel_size=(3, 3), padding="SAME")
    params = conv.init(jax.random.key(0), x)
    count = step_flops(lambda p, x: conv.apply(p, x), params, x)
    # Per output element: 3x3 window x 3 in-channels multiply-adds.
    assert count.flops == pytest.approx(2 * (2 * 16 * 16 * 8) * (3 * 3 * 3))


def test_grad_of_dense_is_three_matmuls_minus_input_grad() -> None:
    dense = nn.Dense(16)
    x = jnp.ones((8, 32))
    params = dense.init(jax.random.key(0), x)

    def loss(p: Variables) -> jax.Array:
        return jnp.sum(jnp.asarray(dense.apply(p, x)))

    # Forward + weight gradient; the input needs no gradient.
    assert step_flops(jax.grad(loss), params).flops == 2 * (2 * 8 * 32 * 16)


def test_param_stats_groups_modules_and_counts_adam() -> None:
    class Two(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array) -> jax.Array:
            return nn.Dense(4)(nn.Dense(8)(x))

    params = Two().init(jax.random.key(0), jnp.ones((1, 3)))
    stats = param_stats(params, optax.adam(1e-3).init(params))
    assert stats.count == (3 * 8 + 8) + (8 * 4 + 4)
    assert stats.bytes == stats.count * 4
    # Linen numbers layers in construction order: the outer Dense(4) is built
    # first, so it is Dense_0.
    assert {m.path: m.count for m in stats.modules} == {
        "params/Dense_0": 36,
        "params/Dense_1": 32,
    }
    # Adam: two moments the size of the params, plus a step count.
    assert stats.optimizer_bytes == pytest.approx(2 * stats.bytes, abs=16)
    assert np.isfinite(stats.optimizer_bytes)


@attrs.frozen
class _FakeDevice:
    stats: dict[str, int] | None

    def memory_stats(self) -> dict[str, int] | None:
        return self.stats


def test_device_memory_reports_the_busiest_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [
        _FakeDevice(
            {
                "bytes_in_use": 10**9,
                "peak_bytes_in_use": 2 * 10**9,
                "bytes_limit": 8 * 10**9,
            }
        ),
        _FakeDevice(
            {
                "bytes_in_use": 3 * 10**9,
                "peak_bytes_in_use": 4 * 10**9,
                "bytes_limit": 8 * 10**9,
            }
        ),
    ]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    assert dict(stats.device_memory()) == {
        "device/memory_in_use_gb": 3.0,
        "device/memory_peak_gb": 4.0,
        "device/memory_limit_gb": 8.0,
    }


def test_device_memory_is_empty_without_allocator_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jax, "local_devices", lambda: [_FakeDevice(None)])
    assert stats.device_memory() == []


def test_step_bytes_counts_memory_traffic() -> None:
    def f(x: jax.Array) -> jax.Array:
        return x * 2.0

    small = stats.step_bytes(f, jnp.ones((1024,), jnp.float32))
    large = stats.step_bytes(f, jnp.ones((1024 * 64,), jnp.float32))
    assert small is not None and large is not None
    # Reads the input and writes the output: ~8 bytes per element.
    assert large == pytest.approx(64 * small, rel=0.1)


def test_peak_bandwidth_knows_our_gpus() -> None:
    assert stats.peak_bandwidth("NVIDIA L4") == 300.0
    assert stats.peak_bandwidth("NVIDIA GeForce RTX 4090") == 1008.0
    assert stats.peak_bandwidth("cpu") is None
