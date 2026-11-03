import csv
import pathlib

import yaml

from core.config import ExperimentConfig, LoopConfig
from core.export import CsvWriter, run_writer, to_plain
from core.optim import AdamW, Constant, OptimConfig


def test_config_yaml_names_union_choices(tmp_path: pathlib.Path) -> None:
    cfg = ExperimentConfig(
        task=LoopConfig(num_steps=10),
        loop=LoopConfig(num_steps=10),
        optim=OptimConfig(optimizer=AdamW(lr=Constant(value=1e-3))),
        workdir=tmp_path,
    )
    run_writer(cfg).flush()
    loaded = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert loaded == to_plain(cfg)
    assert loaded["optim"]["optimizer"]["kind"] == "AdamW"
    assert loaded["optim"]["optimizer"]["lr"]["value"] == 1e-3
    assert loaded["loop"]["inner_loop"] == "SCAN"


def test_csv_is_sparse_and_survives_resume(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "metrics.csv"
    writer = CsvWriter.create(path)
    writer.write_scalars(10, [("train/loss", 2.0)])
    writer.write_scalars(20, [("train/loss", 1.0)])
    writer.write_scalars(20, [("eval/acc", 0.5)])
    writer.flush()

    resumed = CsvWriter.create(path)
    resumed.write_scalars(30, [("train/loss", 0.5)])
    resumed.flush()

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["step"] for r in rows] == ["10", "20", "30"]
    assert rows[0] == {"step": "10", "eval/acc": "", "train/loss": "2"}
    assert rows[1]["eval/acc"] == "0.5"
    assert rows[2]["train/loss"] == "0.5"
