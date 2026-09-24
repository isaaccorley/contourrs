#!/usr/bin/env bash
# Run from any directory; snapshots and build outputs stay in a temporary folder.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
work="$(mktemp -d "${TMPDIR:-/tmp}/contourrs-maintenance.XXXXXX")"
case_name="${1:-noise}"
size="${2:-256}"
mode="${3:-timing}"
features=()
if [[ "$mode" == heap ]]; then features=(--features measure-heap); fi
for version in baseline current unindexed; do
  mkdir -p "$work/$version"
  git -C "$repo" archive 6727cb73529bede6ebb1eeb0445d06964f687d1f | tar -x -C "$work/$version"
  for manifest in Cargo.toml Cargo.lock crates/contourrs/Cargo.toml crates/contourrs-python/Cargo.toml; do
    cp "$repo/$manifest" "$work/$version/$manifest"
  done
  if [[ "$version" != baseline ]]; then
    cp -R "$repo/crates/contourrs/src/." "$work/$version/crates/contourrs/src/"
  fi
  python3 - "$work/$version" "$version" <<'PYTHON'
from pathlib import Path
import sys
root = Path(sys.argv[1])
manifest = root / 'crates/contourrs/Cargo.toml'
manifest.write_text(manifest.read_text().replace('[features]', '[features]\nmeasure-heap = []'))
if sys.argv[2] == 'unindexed':
    source = root / 'crates/contourrs/src/contour_geometry.rs'
    text = source.read_text()
    condition = 'exteriors.len() > 16 && !holes.is_empty()'
    assert condition in text, 'Update the unindexed comparison for the current source.'
    source.write_text(text.replace(condition, 'false'))
PYTHON
  mkdir -p "$work/$version/crates/contourrs/examples"
  cp "$here/maintenance.rs" "$work/$version/crates/contourrs/examples/maintenance.rs"
  CARGO_TARGET_DIR="$work/$version-target" cargo build --release --locked \
    --manifest-path "$work/$version/Cargo.toml" -p contourrs --example maintenance "${features[@]}"
  printf '%s,' "$version"
  RAYON_NUM_THREADS=4 "$work/$version-target/release/examples/maintenance" "$case_name" "$size"
done
printf 'Snapshots preserved in %s\n' "$work"
