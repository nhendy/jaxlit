"""Download a handful of real scenarios for each dataset.

The three differ sharply in how reachable they are, and the script is honest
about that rather than pretending to a uniform interface:

    argoverse  anonymous HTTPS, per scenario (~220 KB each) -- fully automatic
    nuscenes   anonymous HTTPS, but only as one 3.9 GB mini tarball
    womd       auth-gated; prints the gcloud steps and stops

Committed fixtures under `data/testdata/` are what CI runs on, so this script is
for filling in real data locally when you want the opt-in `realdata` tests.

    uv run python scripts/fetch_samples.py --dataset argoverse --count 3
"""

import argparse
import pathlib
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

AV2_BUCKET = "https://s3.amazonaws.com/argoverse"
AV2_PREFIX = "datasets/av2/motion-forecasting"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

NUSCENES_MINI = "https://www.nuscenes.org/data/v1.0-mini.tgz"

WOMD_INSTRUCTIONS = """\
WOMD cannot be downloaded anonymously -- the bucket returns HTTP 401
("Anonymous caller does not have storage.objects.list access"). It is a plain
auth gate, not requester-pays, so no GCP project or billing setup is needed --
just an identity that has accepted the licence:

  1. brew install --cask gcloud-cli
  2. accept the licence at https://waymo.com/open with a Google account
  3. gcloud auth login
  4. gcloud storage cp \\
       gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/validation/validation.tfrecord-00000-of-00150 \\
       data/testdata/womd/

Prefer the smaller `validation_interactive` split over `training`.

The licence forbids redistribution, so real WOMD records must never be
committed. The committed fixture is a synthetic Scenario proto built with the
vendored _pb2 stubs -- it exercises the identical parse path.
"""


def _get(url: str, timeout: int = 60) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} for {url}") from error


def list_av2_scenarios(split: str, count: int) -> list[str]:
    """Scenario ids from the public bucket listing. No credentials needed."""
    url = (
        f"{AV2_BUCKET}/?list-type=2&prefix={AV2_PREFIX}/{split}/"
        f"&delimiter=/&max-keys={count}"
    )
    root = ET.fromstring(_get(url))
    prefixes = root.findall("s3:CommonPrefixes/s3:Prefix", _S3_NS)
    return [p.text.rstrip("/").rsplit("/", 1)[-1] for p in prefixes if p.text]


def fetch_av2(out_dir: pathlib.Path, split: str, count: int) -> None:
    scenario_ids = list_av2_scenarios(split, count)
    if not scenario_ids:
        raise RuntimeError(f"no scenarios listed under {split!r}")

    for scenario_id in scenario_ids:
        target = out_dir / scenario_id
        target.mkdir(parents=True, exist_ok=True)
        base = f"{AV2_BUCKET}/{AV2_PREFIX}/{split}/{scenario_id}"
        for name in (
            f"scenario_{scenario_id}.parquet",
            f"log_map_archive_{scenario_id}.json",
        ):
            path = target / name
            if path.exists():
                print(f"  have {name}")
                continue
            path.write_bytes(_get(f"{base}/{name}"))
            print(f"  got  {name} ({path.stat().st_size // 1024} KB)")
    print(f"{len(scenario_ids)} scenario(s) under {out_dir}")


def fetch_nuscenes(out_dir: pathlib.Path) -> None:
    """The mini split is public, but it is one 3.9 GB tarball -- no partial fetch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "v1.0-mini.tgz"
    if target.exists():
        print(f"already have {target}")
        return
    print(f"downloading {NUSCENES_MINI} (~3.9 GB) -> {target}")
    print("this is the smallest public nuScenes unit; there is no per-scene download.")
    urllib.request.urlretrieve(NUSCENES_MINI, target)
    print(f"done: {target.stat().st_size // (1024 * 1024)} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=("argoverse", "nuscenes", "womd")
    )
    parser.add_argument("--count", type=int, default=3, help="argoverse only")
    parser.add_argument("--split", default="val", help="argoverse only")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="defaults to data/testdata/<dataset>",
    )
    args = parser.parse_args(argv)

    out_dir = args.out or pathlib.Path("data/testdata") / args.dataset

    if args.dataset == "argoverse":
        fetch_av2(out_dir, args.split, args.count)
    elif args.dataset == "nuscenes":
        fetch_nuscenes(out_dir)
    else:
        print(WOMD_INSTRUCTIONS)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
