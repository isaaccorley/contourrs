#!/usr/bin/env bash
set -euo pipefail

mkdir -p docs/tutorials

for nb in examples/quickstart.ipynb examples/dem_contour.ipynb examples/cdl_tiled_polygonize.ipynb; do
  echo "Executing and converting $nb ..."
  uv run --extra all jupyter nbconvert \
    --to markdown \
    --execute \
    --ExecutePreprocessor.timeout=1200 \
    --output-dir docs/tutorials \
    "$nb"
done

# torchgeo notebook requires torch+torchgeo; convert without executing
echo "Converting examples/torchgeo_ftw_polygonize.ipynb (no execute) ..."
uv run --extra all jupyter nbconvert \
  --to markdown \
  --output-dir docs/tutorials \
  examples/torchgeo_ftw_polygonize.ipynb

echo "Done. Run 'uv run --with zensical zensical serve' to preview."
