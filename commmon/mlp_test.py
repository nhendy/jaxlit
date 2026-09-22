import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from commmon.fixtures import parameterized_fixture
from commmon.mlp import MLPConfig

BATCH, FEATURES = 8, 5


@parameterized_fixture
def mlp_config() -> dict[str, MLPConfig]:
    return {
        f"{norm}-{activation}": MLPConfig(
            units=(3, 4, 2), norm=norm, activation=activation
        )
        for norm in ("batch", "layer")
        for activation in ("relu", "gelu", "linear")
    }


@pytest.mark.parametrize("train", [False, True])
def test_mlp_is_elementwise(mlp_config: MLPConfig, train: bool) -> None:
    """Each output row depends only on the input row it came from.

    Dense, LayerNorm and the activations all act along the feature axis, so the
    batch axis is only ever mapped over: running the whole batch at once must
    agree with running each row on its own.
    """
    if train and mlp_config.norm == "batch":
        # BatchNorm normalizes with statistics taken across the batch, so in
        # train mode it couples rows by design. Only its running-average path
        # (train=False) is elementwise.
        pytest.skip("BatchNorm mixes the batch in train mode")

    model = mlp_config.make()
    x = jax.random.normal(jax.random.key(0), (BATCH, FEATURES))
    variables = model.init(jax.random.key(1), x, train=train)

    batched = model.apply(variables, x, train=train)
    one_at_a_time = jnp.concatenate(
        [model.apply(variables, x[i : i + 1], train=train) for i in range(BATCH)]
    )

    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    # `uv run python -m commmon.mlp_test [pytest args]` runs just this file.
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
