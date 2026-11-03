from typing import Literal

import attrs
import flax.linen as nn
import jax
import jaxtyping as jt


@attrs.frozen
class MLPConfig:
    """A simple MLP model."""

    # Number of units per layer
    units: tuple[int, ...]
    # Normalization
    norm: Literal["batch", "layer"]
    # Activation
    activation: Literal["relu", "gelu", "linear"]

    def make(self, dtype: jt.DTypeLike | None = None) -> "MLP":
        return MLP(self, dtype=dtype)


class MLP(nn.Module):
    config: MLPConfig
    # Compute dtype of the layers (params stay float32). A module attribute,
    # not a call argument: flax's `tabulate` traces call arguments as arrays.
    dtype: jt.DTypeLike | None = None

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        dtype = self.dtype
        for i, unit in enumerate(self.config.units):
            x = nn.Dense(
                features=unit, use_bias=self.config.norm != "batch", dtype=dtype
            )(x)
            if i != len(self.config.units) - 1:
                match self.config.norm:
                    case "batch":
                        x = nn.BatchNorm(use_running_average=not train, dtype=dtype)(x)
                    case "layer":
                        x = nn.LayerNorm(dtype=dtype)(x)
                    case _:
                        raise ValueError(f"Unknown norm {self.norm}")

                match self.config.activation:
                    case "relu":
                        x = nn.relu(x)
                    case "gelu":
                        x = nn.gelu(x)
                    case "linear":
                        pass
                    case _:
                        raise ValueError(f"Unknown activation {self.config.activation}")
        return x
