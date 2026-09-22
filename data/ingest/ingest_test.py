"""Ingest writer and CLI tests.

The CLI is exercised through a subprocess rather than by calling `main()`:
`app.run()` and flag parsing are part of what this entrypoint exists to do, and
the Grain-flags problem it sidesteps only shows up when absl actually runs.
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

from commmon.fixtures import parameterized_fixture
from data.ingest.writer import WRITER_OPTIONS, read_shards, shard_paths, write_shards
from data.schema import decode_scenario_id
from data.specs import ALL_SPECS, DatasetSpec
from data.testing import synthetic_scenario

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AV2_TESTDATA = REPO_ROOT / "data" / "testdata" / "argoverse"
NUM_SCENARIOS = 7


@parameterized_fixture
def spec() -> dict[str, DatasetSpec]:
    return dict(ALL_SPECS)


@pytest.fixture
def scenarios(spec: DatasetSpec) -> list:
    rng = np.random.default_rng(0)
    return [
        synthetic_scenario(spec, 5, 9, rng, scenario_id=f"{spec.name}-{i:03d}")
        for i in range(NUM_SCENARIOS)
    ]


# --- writer -------------------------------------------------------------------


def test_shard_paths_are_canonical(tmp_path: pathlib.Path) -> None:
    paths = shard_paths(tmp_path, 3, "av2")
    assert [p.name for p in paths] == [
        "av2-00000-of-00003.arrayrecord",
        "av2-00001-of-00003.arrayrecord",
        "av2-00002-of-00003.arrayrecord",
    ]


def test_group_size_is_one() -> None:
    """Random access is the reason this library uses ArrayRecord at all.

    The default groups records together, so reading one decompresses the whole
    group -- which would quietly undo the MapDataset choice.
    """
    assert WRITER_OPTIONS == "group_size:1"


def test_round_trip_preserves_every_scenario(
    scenarios: list, tmp_path: pathlib.Path, spec: DatasetSpec
) -> None:
    paths = write_shards(scenarios, tmp_path, num_shards=3, prefix=spec.name)
    restored = list(read_shards(paths))
    assert len(restored) == NUM_SCENARIOS
    assert {decode_scenario_id(s.scenario_id) for s in restored} == {
        f"{spec.name}-{i:03d}" for i in range(NUM_SCENARIOS)
    }


def test_scenarios_are_spread_round_robin(
    scenarios: list, tmp_path: pathlib.Path
) -> None:
    """Contiguous chunking would preserve the raw datasets' ordering inside each
    shard; round-robin breaks it."""
    paths = write_shards(scenarios, tmp_path, num_shards=3)
    counts = [len(list(read_shards([p]))) for p in paths]
    assert sum(counts) == NUM_SCENARIOS
    assert max(counts) - min(counts) <= 1


def test_writer_creates_missing_directories(
    scenarios: list, tmp_path: pathlib.Path
) -> None:
    nested = tmp_path / "a" / "b"
    write_shards(scenarios, nested, num_shards=2)
    assert nested.is_dir()


def test_writer_closes_shards_when_the_source_raises(
    tmp_path: pathlib.Path, spec: DatasetSpec
) -> None:
    """A half-written shard that was never closed fails confusingly at read time,
    so the writer closes in a finally block."""

    def exploding():
        yield synthetic_scenario(spec, 3, 5)
        raise RuntimeError("source blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        write_shards(exploding(), tmp_path, num_shards=2)

    # The first record still landed and the file is readable, not truncated.
    written = sorted(tmp_path.glob("*.arrayrecord"))
    assert len(list(read_shards(written))) == 1


def test_writer_rejects_zero_shards(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="num_shards"):
        write_shards([], tmp_path, num_shards=0)


def test_empty_source_writes_empty_shards(tmp_path: pathlib.Path) -> None:
    paths = write_shards([], tmp_path, num_shards=2)
    assert len(list(read_shards(paths))) == 0


# --- CLI ----------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "data.ingest", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


requires_av2 = pytest.mark.skipif(
    not AV2_TESTDATA.exists(), reason="AV2 fixtures missing"
)


@requires_av2
def test_cli_ingests_real_data(tmp_path: pathlib.Path) -> None:
    result = _run_cli(
        "--dataset=av2",
        f"--src={AV2_TESTDATA}",
        f"--out={tmp_path}",
        "--num_shards=2",
    )
    assert result.returncode == 0, result.stderr
    assert "verified 3 records" in result.stderr
    assert len(list(read_shards(sorted(tmp_path.glob("*.arrayrecord"))))) == 3


@requires_av2
def test_cli_limit_stops_early(tmp_path: pathlib.Path) -> None:
    result = _run_cli(
        "--dataset=av2",
        f"--src={AV2_TESTDATA}",
        f"--out={tmp_path}",
        "--num_shards=1",
        "--limit=2",
    )
    assert result.returncode == 0, result.stderr
    assert len(list(read_shards(sorted(tmp_path.glob("*.arrayrecord"))))) == 2


def test_cli_refuses_object_store_paths_with_a_reason(tmp_path: pathlib.Path) -> None:
    """gs:// is refused because TF (which epath needs for it) cannot coexist with
    array_record in one process. The message must say that and give the way out."""
    result = _run_cli(
        "--dataset=womd",
        "--src=gs://waymo_open_dataset_motion_v_1_2_0/nope",
        f"--out={tmp_path / 'out'}",
    )
    assert result.returncode != 0
    assert "segfault" in result.stderr and "gcloud storage cp" in result.stderr


def test_cli_rejects_a_missing_source(tmp_path: pathlib.Path) -> None:
    result = _run_cli(
        "--dataset=av2", f"--src={tmp_path / 'nope'}", f"--out={tmp_path / 'out'}"
    )
    assert result.returncode != 0
    assert "not a directory" in result.stderr


def test_cli_rejects_an_unknown_dataset(tmp_path: pathlib.Path) -> None:
    result = _run_cli(
        "--dataset=kitti", f"--src={tmp_path}", f"--out={tmp_path / 'out'}"
    )
    assert result.returncode != 0


def test_cli_names_the_unwritten_adapter(tmp_path: pathlib.Path) -> None:
    """An unwritten adapter should say which one and what it needs, not ImportError.

    Only nuScenes remains; av2 and womd are implemented.
    """
    result = _run_cli(
        "--dataset=nuscenes", f"--src={tmp_path}", f"--out={tmp_path / 'out'}"
    )
    assert result.returncode != 0
    assert "step 6" in result.stderr


def test_cli_womd_reports_a_missing_shard_clearly(tmp_path: pathlib.Path) -> None:
    """WOMD is implemented now, so an empty dir is a data problem, not a
    not-implemented one -- and the message should name the expected filenames."""
    result = _run_cli(
        "--dataset=womd", f"--src={tmp_path}", f"--out={tmp_path / 'out'}"
    )
    assert result.returncode != 0
    assert "no TFRecord shards" in result.stderr
