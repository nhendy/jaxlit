import attrs
import optax
from attrs import validators as v

from commmon.mlp import MLPConfig
from core.config import ExperimentConfig, LoopConfig
from core.model import LinenModel
from core.optim import AdamW, OptimConfig, WarmupCosine
from tasks.mnist.task import Classifier, MnistTask


@attrs.frozen(kw_only=True)
class MnistConfig:
    mlp: MLPConfig = MLPConfig(units=(512, 256, 10), norm="layer", activation="relu")
    batch_size: int = attrs.field(default=128, validator=v.gt(0))
    eval_batch_size: int = attrs.field(default=1000, validator=v.gt(0))

    def make(self, tx: optax.GradientTransformation) -> MnistTask:
        return MnistTask(model=LinenModel(module=Classifier(mlp=self.mlp)), tx=tx)


def default(workdir: str, num_steps: int = 3000) -> ExperimentConfig[MnistConfig]:
    return ExperimentConfig(
        task=MnistConfig(),
        loop=LoopConfig(num_steps=num_steps, steps_per_call=10),
        optim=OptimConfig(
            optimizer=AdamW(
                lr=WarmupCosine(
                    peak=2e-3,
                    warmup_steps=min(200, num_steps // 10),
                    decay_steps=num_steps,
                )
            )
        ),
        workdir=workdir,
    )
