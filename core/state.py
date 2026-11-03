"""The default train state. Tasks may use their own; nothing in `core` requires this."""

import dataclasses
from typing import Self, cast

import jax
import jax.numpy as jnp
import optax
from flax import struct

from core.types import PRNGKey


@struct.dataclass
class TrainState[ParamsT, ExtraT]:
    """Everything a train step reads and writes.

    `step` counts `train_step` calls, which is what the loop resumes from. For
    most tasks that is one optimizer update; a task doing several updates per
    step (PPO) uses `update_params` for the inner ones and `advance` once.
    """

    step: jax.Array
    params: ParamsT
    opt_state: optax.OptState
    rng: PRNGKey
    # Task-specific state that is not trained: EMA params, env state, ...
    extra: ExtraT

    @classmethod
    def create(
        cls,
        *,
        params: ParamsT,
        tx: optax.GradientTransformation,
        rng: PRNGKey,
        extra: ExtraT,
    ) -> Self:
        return cls(
            step=jnp.zeros((), jnp.int32),
            params=params,
            opt_state=tx.init(cast(optax.Params, params)),
            rng=rng,
            extra=extra,
        )

    def update_params(self, tx: optax.GradientTransformation, grads: ParamsT) -> Self:
        """One optimizer update, without counting a step."""
        # Any pytree is a valid optax tree; optax's annotation just can't say so.
        params = cast(optax.Params, self.params)
        updates, opt_state = tx.update(
            cast(optax.Updates, grads), self.opt_state, params
        )
        new_params = cast(ParamsT, optax.apply_updates(params, updates))
        return dataclasses.replace(self, params=new_params, opt_state=opt_state)

    def advance(self) -> Self:
        return dataclasses.replace(self, step=self.step + 1)

    def apply_gradients(self, tx: optax.GradientTransformation, grads: ParamsT) -> Self:
        """One optimizer update and one step: the usual train step."""
        return self.update_params(tx, grads).advance()

    def next_rng(self) -> tuple[Self, PRNGKey]:
        """Split off a fresh key, threading the rest through the state."""
        rng, key = jax.random.split(self.rng)
        return dataclasses.replace(self, rng=rng), key
