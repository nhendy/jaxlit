"""Vendored protobuf stubs for the Waymo Open Motion Dataset.

`waymo-open-dataset` publishes manylinux x86_64 wheels only -- there is no
arm64 macOS build -- and it would pull in TensorFlow, which cannot share a
process with `array_record` on this platform (both are C++ extensions; loading
both segfaults). Since the motion dataset is just `Scenario` protos in TFRecord
framing, vendoring the generated stubs removes the dependency entirely.

The two `.proto` sources are committed alongside the stubs so the generation is
reproducible:

    protoc -I . --python_out=<out> waymo_open_dataset/protos/{scenario,map}.proto

The generated stubs then need one fixup: protoc emits absolute imports
(`from waymo_open_dataset.protos import map_pb2`) which do not resolve once the
tree is vendored, so they are rewritten to this package path. Also note protoc
and the protobuf runtime are version-coupled -- the runtime must be >= the
gencode version protoc emits, which is why pyproject pins protobuf>=7.36.1.

`scenario.proto` is trimmed from upstream: fields 12 and 13
(`compressed_frame_laser_data`, `frame_camera_tokens`) and their imports are
removed, which drops four transitive proto dependencies we have no use for.
Both are marked `reserved`, so records that contain them still parse -- the
sensor payloads simply land in the unknown-field set and are discarded.

Regenerate when Waymo revs the format: re-fetch from
github.com/waymo-research/waymo-open-dataset under src/, re-apply the trim,
re-run protoc.
"""

from data.sources._womd_proto.waymo_open_dataset.protos import (  # noqa: F401
    map_pb2,
    scenario_pb2,
)
