"""Weights & Biases as one more `MetricWriter`.

Everything that logs -- `LogMetrics`, `Evaluate`, a task's own callbacks such
as `tasks.flow.sampler.SampleGrid` -- already talks to a `MetricWriter`, so W&B
needs no callback of its own: `run_writer(cfg, wandb=Wandb(project=...))` adds
this writer next to TensorBoard and the mlcompare CSV, and every scalar and
image reaches all three.

    python -m tasks.mnist.main --workdir=... --wandb_project=jaxlit
    python -m tasks.mnist.main --workdir=... --wandb_project=jaxlit --wandb_mode=offline
"""

import enum
import pathlib
from collections.abc import Sequence
from typing import Any

import attrs
import numpy as np
import wandb
from absl import flags

from core.writers import Figures, Images, Scalars, to_uint8

# The id of the W&B run that logs this workdir. Kept beside the checkpoints so
# a resumed job appends to the same run instead of starting a new one.
RUN_ID_FILE = "wandb_run_id"


class Mode(enum.Enum):
    ONLINE = "online"
    # Logs to <workdir>/wandb; upload later with `wandb sync`.
    OFFLINE = "offline"
    # Every call is a no-op. For tests and dry runs.
    DISABLED = "disabled"


@attrs.frozen(kw_only=True)
class Wandb:
    project: str
    entity: str | None = None
    # Defaults to the workdir's name.
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: Mode = Mode.ONLINE


@attrs.frozen
class WandbWriter:
    """A `MetricWriter` over one W&B run.

    The run is a handle to an external service, held the same way
    `TensorBoardWriter` holds its event file: this object never changes, the
    service's state does.
    """

    run: Any

    @classmethod
    def create(
        cls, options: Wandb, workdir: pathlib.Path, config: dict[str, Any]
    ) -> "WandbWriter":

        workdir.mkdir(parents=True, exist_ok=True)
        id_file = workdir / RUN_ID_FILE
        run_id = id_file.read_text().strip() if id_file.exists() else None
        run = wandb.init(
            project=options.project,
            entity=options.entity,
            name=options.name or workdir.name,
            group=options.group,
            tags=list(options.tags),
            config=config,
            dir=str(workdir),
            mode=options.mode.value,
            id=run_id,
            resume="allow",
        )
        if options.mode is not Mode.DISABLED:
            id_file.write_text(run.id)
        return cls(run)

    def write_scalars(self, step: int, scalars: Scalars) -> None:
        self.run.log(dict(scalars), step=step)

    def write_images(self, step: int, images: Images) -> None:

        self.run.log(
            {name: wandb.Image(_for_wandb(image)) for name, image in images},
            step=step,
        )

    def write_figures(self, step: int, figures: Figures) -> None:

        # wandb.Plotly keeps the figure interactive: zoom, pan and hover in
        # the W&B panel, not a screenshot.
        self.run.log(
            {name: wandb.Plotly(figure) for name, figure in figures}, step=step
        )

    def flush(self) -> None:
        # W&B streams in the background; `finish` runs at interpreter exit.
        return None


def _for_wandb(image: np.ndarray) -> np.ndarray:
    pixels = to_uint8(image)
    return pixels[..., 0] if pixels.shape[-1] == 1 else pixels


# --- flags ------------------------------------------------------------------

_PROJECT = flags.DEFINE_string(
    "wandb_project", None, "Log to this W&B project. Unset: no W&B."
)
_ENTITY = flags.DEFINE_string("wandb_entity", None, "W&B team or user.")
_GROUP = flags.DEFINE_string("wandb_group", None, "W&B group, e.g. a sweep name.")
_TAGS = flags.DEFINE_list("wandb_tags", [], "W&B tags.")
_MODE = flags.DEFINE_enum_class("wandb_mode", Mode.ONLINE, Mode, "W&B mode.")


def from_flags() -> Wandb | None:
    if _PROJECT.value is None:
        return None
    tags: Sequence[str] = _TAGS.value
    return Wandb(
        project=_PROJECT.value,
        entity=_ENTITY.value,
        group=_GROUP.value,
        tags=tuple(tags),
        mode=_MODE.value,
    )
