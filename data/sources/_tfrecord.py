"""Minimal TFRecord reader -- deliberately without TensorFlow.

TFRecord is a trivially simple container, and reading it ourselves avoids a
dependency that actively breaks this project: TensorFlow and `array_record`
are both C++ extensions that cannot be loaded into the same process on macOS
arm64 (either import order segfaults). Since ingest must write ArrayRecord,
TensorFlow can never be present, so `grain.experimental.TFRecordIterDataset`
and `tf.data.TFRecordDataset` are both unavailable.

The framing, little-endian, per record::

    uint64  length
    uint32  masked crc32c of the length bytes
    bytes   payload[length]
    uint32  masked crc32c of the payload

Checksums are skipped rather than verified. They guard against bit rot in
transit, and every path that matters here already has its own integrity check:
`gcloud storage cp` verifies a CRC32C on download, and decoding a record
re-runs the full eincheck schema. Verifying here would mean either a pure-Python
CRC32C (slow over hundreds of GB) or a new dependency, for a third check.
"""

import struct
from collections.abc import Iterator

from etils import epath

_HEADER = struct.Struct("<Q")
_HEADER_BYTES = 12  # 8-byte length + 4-byte CRC
_FOOTER_BYTES = 4


def read_records(path: epath.PathLike, limit: int | None = None) -> Iterator[bytes]:
    """Yield raw record payloads from a TFRecord file.

    A truncated tail ends iteration rather than raising. That is deliberate and
    useful: it makes a byte-range prefix of a remote shard readable, so a few
    records can be pulled for a fixture without fetching a ~270 MB file.
    """
    count = 0
    with epath.Path(path).open("rb") as stream:
        while limit is None or count < limit:
            header = stream.read(_HEADER_BYTES)
            if len(header) < _HEADER_BYTES:
                return  # clean EOF, or a truncated header
            (length,) = _HEADER.unpack(header[:8])
            payload = stream.read(length)
            if len(payload) < length:
                return  # truncated payload -- stop, do not yield a partial record
            stream.read(_FOOTER_BYTES)
            count += 1
            yield payload


def write_records(path: epath.PathLike, payloads: Iterator[bytes]) -> int:
    """Write payloads in TFRecord framing. Returns the record count.

    Used to build test fixtures. The checksum fields are written as zeros, which
    `read_records` ignores -- these files are for our own round-tripping, not for
    feeding to TensorFlow, which would reject them.
    """
    count = 0
    with epath.Path(path).open("wb") as stream:
        for payload in payloads:
            stream.write(_HEADER.pack(len(payload)))
            stream.write(b"\x00" * 4)
            stream.write(payload)
            stream.write(b"\x00" * _FOOTER_BYTES)
            count += 1
    return count
