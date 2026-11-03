"""Train an MNIST classifier.

uv run python -m tasks.mnist.main --workdir=/tmp/jaxlit/mnist
"""

import jax
import orbax.checkpoint as ocp
from absl import app, flags, logging

from core import distributed, launch, tracking
from core.callbacks import (
    AtEnd,
    Checkpoint,
    Evaluate,
    Every,
    LogMetrics,
    evaluate,
    restore_latest,
)
from core.distributed import Partitioner, Replicated
from core.export import run_writer
from core.loop import fit
from tasks.mnist import config as config_lib
from tasks.mnist import data

_WORKDIR = flags.DEFINE_string("workdir", None, "Logs and checkpoints.", required=True)
_NUM_STEPS = flags.DEFINE_integer("num_steps", 3000, "Train steps.")


def main(argv: list[str]) -> None:
    del argv
    distributed.initialize(launch.from_flags())
    launch.shape_checks_from_flags()
    cfg = config_lib.default(_WORKDIR.value, _NUM_STEPS.value)
    task = cfg.task.make(cfg.optim.make())

    train = data.batches(
        data.load(data.Split.TRAIN),
        batch_size=cfg.task.batch_size,
        seed=cfg.loop.seed,
        training=True,
    )
    test = data.batches(
        data.load(data.Split.TEST),
        batch_size=cfg.task.eval_batch_size,
        seed=cfg.loop.seed,
        training=False,
    )

    writer = run_writer(cfg, wandb=tracking.from_flags())
    partitioner = Partitioner.create(cfg.loop.mesh)
    sharding = Replicated()
    manager = ocp.CheckpointManager(cfg.workdir.absolute() / "ckpt")
    example = next(iter(train))
    abstract = partitioner.abstract_state(
        task.init,
        jax.random.key(cfg.loop.seed),
        partitioner.to_global(example),
        sharding,
    )

    state = fit(
        task,
        config=cfg.loop,
        train=train,
        state=restore_latest(manager, abstract),
        partitioner=partitioner,
        sharding=sharding,
        callbacks=(
            LogMetrics(
                trigger=Every(n=100),
                writer=writer,
                learning_rate=cfg.optim.schedule(),
            ),
            Evaluate(
                trigger=Every(n=1000),
                task=task,
                batches=test,
                writer=writer,
                partitioner=partitioner,
            ),
            Checkpoint(trigger=AtEnd(), manager=manager),
        ),
    )
    result = evaluate(task, state, test, partitioner=partitioner)
    logging.info(
        "test accuracy %.4f, loss %.4f",
        float(result.accuracy.compute()),
        float(result.loss.compute()),
    )


if __name__ == "__main__":
    app.run(main)
