"""Runs on disk in the format `~/dev/mlcompare` reads.

mlcompare treats any directory holding a `config.yaml` and a `metrics.csv` as a
run, so the workdir becomes one:

    <workdir>/config.yaml    the ExperimentConfig, nested; dotted keys in the UI
    <workdir>/metrics.csv    step,train/loss,eval/model/ade,...  (sparse rows)

Point it at the parent of several workdirs to compare them:
`uv run mlcompare /tmp/jaxlit`.

The CSV is a `MetricWriter`, not a callback: train and eval scalars already
flow through the writer, so this sees every metric without a second path.
"""

import csv
import enum
import itertools
import pathlib
from collections.abc import Iterator
from typing import Any, cast

import attrs
import yaml

from core import distributed, tracking
from core.config import ExperimentConfig
from core.writers import (
    Figures,
    Images,
    LoggingWriter,
    MetricWriter,
    MultiWriter,
    ScalarRecord,
    Scalars,
    TensorBoardWriter,
)

CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.csv"
STEP_COLUMN = "step"
# Names the config's tagged-union choice -- `optim.optimizer.kind: AdamW` -- so
# mlcompare can group by it. Plain field names otherwise.
KIND_KEY = "kind"


def to_plain(value: object) -> object:
    """A config tree as YAML-safe builtins. The I/O boundary: strings start here."""
    if attrs.has(type(value)):
        fields = attrs.fields(type(value))
        plain: dict[str, object] = {KIND_KEY: type(value).__name__}
        plain.update((f.name, to_plain(getattr(value, f.name))) for f in fields)
        return plain
    match value:
        case enum.Enum():
            return value.name
        case pathlib.Path():
            return str(value)
        case tuple() | list():
            return [to_plain(x) for x in value]  # pyright: ignore[reportUnknownVariableType]
        case bool() | int() | float() | str() | None:
            return value
        case _:
            return repr(value)


def write_config(directory: pathlib.Path, config: object) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONFIG_FILE
    path.write_text(yaml.safe_dump(to_plain(config), sort_keys=False))
    return path


def _read(path: pathlib.Path) -> Iterator[ScalarRecord]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row.pop(STEP_COLUMN))
            for name, value in row.items():
                if value:
                    yield ScalarRecord(step=step, name=name, value=float(value))


@attrs.frozen
class CsvWriter:
    """Sparse `step,<metric>...` rows, one per logged step.

    Columns appear as metrics do (eval ones only at the first eval), so the file
    is rewritten whole on `flush` rather than appended to; it is small, and the
    temp-file rename means mlcompare never reads half of one. Rows already on
    disk are loaded first, so a resumed run keeps its history; a step logged
    again overwrites the old value.
    """

    path: pathlib.Path
    records: list[ScalarRecord] = attrs.field(factory=list)

    @classmethod
    def create(cls, path: pathlib.Path) -> "CsvWriter":
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path, list(_read(path)) if path.exists() else [])

    def write_scalars(self, step: int, scalars: Scalars) -> None:
        self.records.extend(
            ScalarRecord(step=step, name=n, value=v) for n, v in scalars
        )

    def write_images(self, step: int, images: Images) -> None:
        return None

    def write_figures(self, step: int, figures: Figures) -> None:
        return None

    def flush(self) -> None:
        columns = sorted({r.name for r in self.records})
        index = {name: i for i, name in enumerate(columns)}
        tmp = self.path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as handle:
            out = csv.writer(handle)
            out.writerow([STEP_COLUMN, *columns])
            ordered = sorted(self.records, key=lambda r: r.step)
            for step, group in itertools.groupby(ordered, key=lambda r: r.step):
                row: list[str] = [""] * len(columns)
                for record in group:  # later writes win
                    row[index[record.name]] = f"{record.value:.7g}"
                out.writerow([str(step), *row])
        tmp.replace(self.path)


def run_writer(
    config: ExperimentConfig[Any], *, wandb: "tracking.Wandb | None" = None
) -> MultiWriter:
    """Log lines, TensorBoard under `<workdir>/logs`, and an mlcompare run at
    `<workdir>`. Writes the config now, so the run shows up before step one.

    `wandb` adds a W&B run with the same config. Like the files, it comes from
    the primary process only; the others just log, so N hosts
    sharing a workdir do not race on one CSV.
    """
    if not distributed.is_primary():
        return MultiWriter((LoggingWriter(),))
    write_config(config.workdir, config)
    writers: list[MetricWriter] = [
        LoggingWriter(),
        TensorBoardWriter.create(config.workdir / "logs"),
        CsvWriter.create(config.workdir / METRICS_FILE),
    ]
    if wandb is not None:
        writers.append(
            tracking.WandbWriter.create(
                wandb, config.workdir, cast(dict[str, Any], to_plain(config))
            )
        )
    return MultiWriter(tuple(writers))
