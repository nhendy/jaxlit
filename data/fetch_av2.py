"""Download AV2 motion-forecasting splits from Argoverse's public S3 bucket.

    uv run python -m data.fetch_av2 --split=val --out=/data/raw/val

Standard library only: the bucket allows anonymous reads over HTTPS, so this
is the S3 list API plus parallel GETs -- no AWS CLI, no credentials, no
third-party binary. Each scenario is two small objects (a parquet and a map
JSON), so throughput comes from concurrency, not from big transfers.

Resumable: an object already on disk at the listed size is skipped.

Parallel across processes as well as threads: TLS and response handling cost
enough Python per object that one process tops out at one core (~90 scenes/s
from Europe to the bucket in us-east-1), so the object list is split over
`--processes` processes, each with its own thread pool.
"""

import concurrent.futures
import http.client
import itertools
import pathlib
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import attrs
from absl import app, flags, logging

BUCKET_HOST = "argoverse.s3.amazonaws.com"
BUCKET_URL = f"https://{BUCKET_HOST}"
PREFIX = "datasets/av2/motion-forecasting"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

_SPLIT = flags.DEFINE_enum("split", None, ["train", "val", "test"], "AV2 split.")
_OUT = flags.DEFINE_string("out", None, "Directory to download into.")
_THREADS = flags.DEFINE_integer("threads", 64, "Concurrent downloads per process.")
_PROCESSES = flags.DEFINE_integer("processes", 8, "Download processes.")
_LIMIT = flags.DEFINE_integer("limit", None, "Stop after this many objects.")


@attrs.frozen(kw_only=True)
class S3Object:
    key: str
    size: int


def list_objects(prefix: str) -> Iterator[S3Object]:
    """Every object under `prefix`, following list continuation tokens."""
    token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token is not None:
            query["continuation-token"] = token
        url = f"{BUCKET_URL}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.iter(f"{_NS}Contents"):
            yield S3Object(
                key=item.findtext(f"{_NS}Key", ""),
                size=int(item.findtext(f"{_NS}Size", "0")),
            )
        if root.findtext(f"{_NS}IsTruncated") != "true":
            return
        token = root.findtext(f"{_NS}NextContinuationToken")


# One kept-alive HTTPS connection per download thread. `urlopen` opens a new
# connection per request, and from Europe the TCP + TLS handshake to the
# bucket's us-east-1 endpoint costs ~0.7 s -- per object, of ~450k objects.
_local = threading.local()


def _connection() -> http.client.HTTPSConnection:
    conn: http.client.HTTPSConnection | None = getattr(_local, "conn", None)
    if conn is None:
        conn = http.client.HTTPSConnection(BUCKET_HOST, timeout=120)
        _local.conn = conn
    return conn


def _get(key: str) -> bytes:
    conn = _connection()
    try:
        conn.request("GET", "/" + urllib.parse.quote(key))
        response = conn.getresponse()
        body = response.read()
    except (OSError, http.client.HTTPException):
        conn.close()
        _local.conn = None
        raise
    if response.status != 200:
        raise OSError(f"GET {key}: HTTP {response.status}")
    return body


def download(obj: S3Object, dest: pathlib.Path, retries: int = 5) -> int:
    """Fetch one object; returns bytes written (0 if already present)."""
    if dest.exists() and dest.stat().st_size == obj.size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(_get(obj.key))
            tmp.replace(dest)
            return obj.size
        except (OSError, http.client.HTTPException):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return 0


def _download_all(
    objects: list[S3Object], prefix: str, out: pathlib.Path, threads: int
) -> int:
    """One process's share: download `objects` with a thread pool; bytes written."""
    total = 0
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        futures = [
            pool.submit(download, obj, out / obj.key[len(prefix) :]) for obj in objects
        ]
        for future in concurrent.futures.as_completed(futures):
            total += future.result()
    return total


def fetch(
    split: str, out: pathlib.Path, threads: int, processes: int, limit: int | None
) -> int:
    prefix = f"{PREFIX}/{split}/"
    started = time.perf_counter()
    # Scenario ids are UUIDs, so the 16 hex first characters partition the
    # keys: list them in parallel rather than paging 1000 at a time in series.
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        parts = pool.map(lambda c: list(list_objects(prefix + c)), "0123456789abcdef")
        objects = list(itertools.islice(itertools.chain.from_iterable(parts), limit))
    logging.info(
        "%s: listed %d objects in %.0fs",
        split,
        len(objects),
        time.perf_counter() - started,
    )
    # Strided, so every process gets a mix rather than one contiguous key range.
    shares = [objects[i::processes] for i in range(processes)]
    total = 0
    with concurrent.futures.ProcessPoolExecutor(processes) as pool:
        for written in pool.map(
            _download_all,
            shares,
            [prefix] * processes,
            [out] * processes,
            [threads] * processes,
        ):
            total += written
    elapsed = time.perf_counter() - started
    logging.info(
        "%s: %d objects, %.1f GB in %.0fs (%.0f MB/s)",
        split,
        len(objects),
        total / 1e9,
        elapsed,
        total / max(elapsed, 1e-9) / 1e6,
    )
    return len(objects)


def main(argv: list[str]) -> None:
    del argv
    if _SPLIT.value is None or _OUT.value is None:
        raise app.UsageError("--split and --out are required")
    fetch(
        _SPLIT.value,
        pathlib.Path(_OUT.value).expanduser(),
        _THREADS.value,
        _PROCESSES.value,
        _LIMIT.value,
    )


if __name__ == "__main__":
    app.run(main)
