"""W&B writer in offline mode: no account, no network."""

import pathlib

import numpy as np
import pytest

from core.config import ExperimentConfig, LoopConfig
from core.export import run_writer
from core.optim import Constant, OptimConfig, Sgd
from core.tracking import RUN_ID_FILE, Mode, Wandb, WandbWriter

pytest.importorskip("wandb")


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_SILENT", "true")


def config(workdir: pathlib.Path) -> ExperimentConfig[LoopConfig]:
    return ExperimentConfig(
        task=LoopConfig(num_steps=10),
        loop=LoopConfig(num_steps=10),
        optim=OptimConfig(optimizer=Sgd(lr=Constant(value=0.1))),
        workdir=workdir,
    )


def test_scalars_images_and_config_reach_the_run(tmp_path: pathlib.Path) -> None:
    writer = run_writer(
        config(tmp_path), wandb=Wandb(project="jaxlit-test", mode=Mode.OFFLINE)
    )
    (wb,) = [w for w in writer.writers if isinstance(w, WandbWriter)]
    writer.write_scalars(1, [("train/loss", 2.0)])
    writer.write_scalars(2, [("train/loss", 1.5), ("eval/acc", 0.5)])
    writer.write_images(2, [("samples", np.ones((8, 8, 1), np.float32))])
    # W&B buffers a step's values until a later step is logged -- that is what
    # merges step 2's scalars and image into one row. Step 3 commits step 2.
    writer.write_scalars(3, [("train/other", 0.0)])
    writer.flush()

    assert wb.run.summary["train/loss"] == 1.5
    assert wb.run.summary["eval/acc"] == 0.5
    assert "samples" in wb.run.summary.keys()
    # Nested config, with the union choice named, as in config.yaml.
    assert wb.run.config["optim"]["optimizer"]["kind"] == "Sgd"
    assert (tmp_path / RUN_ID_FILE).read_text() == wb.run.id
    # Still writes TensorBoard and the mlcompare CSV alongside.
    assert (tmp_path / "metrics.csv").exists()
    wb.run.finish()


def test_resume_reuses_the_run_id(tmp_path: pathlib.Path) -> None:
    options = Wandb(project="jaxlit-test", mode=Mode.OFFLINE)
    first = WandbWriter.create(options, tmp_path, {})
    first.run.finish()
    second = WandbWriter.create(options, tmp_path, {})
    assert second.run.id == first.run.id
    second.run.finish()


def test_disabled_is_a_no_op(tmp_path: pathlib.Path) -> None:
    writer = WandbWriter.create(
        Wandb(project="jaxlit-test", mode=Mode.DISABLED), tmp_path, {}
    )
    writer.write_scalars(1, [("x", 1.0)])
    assert not (tmp_path / RUN_ID_FILE).exists()
    writer.run.finish()


def test_figures_are_logged_interactive(tmp_path: pathlib.Path) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    writer = WandbWriter.create(
        Wandb(project="jaxlit-test", mode=Mode.OFFLINE), tmp_path, {}
    )
    writer.write_figures(1, [("fig", go.Figure([go.Scatter(x=[0, 1], y=[0, 1])]))])
    writer.write_scalars(2, [("x", 1.0)])  # commits step 1
    assert "fig" in writer.run.summary.keys()
    writer.run.finish()
