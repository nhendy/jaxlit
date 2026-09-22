#!/usr/bin/env bash
# Convert a raw dataset to ArrayRecord shards and push them to R2.
#
#   scripts/ingest_and_sync.sh av2 /data/av2/train train
#   scripts/ingest_and_sync.sh av2 /data/av2/val   val   --limit=1000
#
# Two stages, deliberately separate:
#
#   1. `python -m data.ingest` writes shards to a local staging directory. The
#      ArrayRecord writer is a C++ extension that writes local files; it does not
#      speak object storage, so staging locally is not a workaround, it is the
#      only way.
#   2. `rclone sync` uploads the staging directory. rclone parallelises,
#      checksums, resumes, and skips objects that already match -- so a failed
#      run is re-runnable without re-uploading what already landed.
#
# Staging is kept, not deleted: re-running the sync after a network failure
# should not mean re-converting 200k scenarios. Delete it yourself when the
# upload is verified.
set -euo pipefail

BUCKET="${R2_BUCKET:-jaxlit-data}"
REMOTE="${R2_REMOTE:-r2}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/jaxlit-ingest}"
SHARDS="${NUM_SHARDS:-256}"

usage() {
    echo "usage: $0 <dataset> <src-dir> <split> [extra --flags for data.ingest]" >&2
    echo "  dataset: av2 | womd | nuscenes" >&2
    exit 2
}
[ $# -ge 3 ] || usage

DATASET="$1"; SRC="$2"; SPLIT="$3"; shift 3

for tool in rclone uv; do
    command -v "$tool" >/dev/null || { echo "error: $tool not on PATH" >&2; exit 1; }
done
[ -d "$SRC" ] || { echo "error: --src is not a directory: $SRC" >&2; exit 1; }

# `rclone about` fails fast on a misconfigured remote, which is much clearer
# than discovering it after an hour of conversion.
rclone lsd "${REMOTE}:" >/dev/null 2>&1 || {
    echo "error: rclone remote '${REMOTE}:' is not configured or unreachable." >&2
    echo "       check ~/.config/rclone/rclone.conf (type=s3, provider=Cloudflare)" >&2
    exit 1
}

STAGE="${STAGE_ROOT}/${DATASET}/${SPLIT}"
DEST="${REMOTE}:${BUCKET}/index/${DATASET}/motion/${SPLIT}"

echo "==> converting  ${SRC}"
echo "    staging to  ${STAGE}"
mkdir -p "$STAGE"
uv run python -m data.ingest \
    --dataset="$DATASET" \
    --src="$SRC" \
    --out="$STAGE" \
    --num_shards="$SHARDS" \
    --prefix="$DATASET" \
    "$@"

BYTES=$(du -sk "$STAGE" | cut -f1)
echo "==> uploading   $((BYTES / 1024)) MB -> ${DEST}"
# --transfers: R2 handles parallel PUTs well and the bottleneck is upstream
# bandwidth, not the API. --checksum avoids re-uploading on a second run.
rclone sync "$STAGE" "$DEST" \
    --transfers=16 \
    --checkers=32 \
    --checksum \
    --progress

echo "==> verifying"
rclone check "$STAGE" "$DEST" --checksum --one-way

cat <<EOF

done.
  local staging : ${STAGE}   (kept -- rm it once you trust the upload)
  remote prefix : ${DEST}
  train with    : --shards=/index/${DATASET}/motion/${SPLIT}
EOF
