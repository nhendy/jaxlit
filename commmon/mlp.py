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

    def make(self) -> "MLP":
        return MLP(self)


class MLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(
        self, x: jax.Array, *, train: bool, dtype: jt.DTypeLike | None = None
    ) -> jax.Array:
        for i, unit in enumerate(self.config.units):
            x = nn.Dense(
                features=unit, use_bias=self.config.norm != "batch", dtype=dtype
            )(x)
            if i != len(self.config.units) - 1:
                match self.config.norm:
                    case "batch":
                        x = nn.BatchNorm(use_running_average=not train)(x)
                    case "layer":
                        x = nn.LayerNorm()(x)
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
