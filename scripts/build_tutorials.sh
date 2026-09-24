#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-docs/tutorials}"
mkdir -p "$output_dir"

for nb in examples/quickstart.ipynb examples/dem_contour.ipynb examples/cdl_tiled_polygonize.ipynb; do
  echo "Executing and converting $nb ..."
  uv run --locked --extra docs jupyter nbconvert \
    --to markdown \
    --execute \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip_ci"]' \
    --ExecutePreprocessor.timeout=1200 \
    --output-dir "$output_dir" \
    "$nb"
done

# torchgeo notebook requires torch+torchgeo; convert without executing
echo "Converting examples/torchgeo_ftw_polygonize.ipynb (no execute) ..."
uv run --locked --extra docs jupyter nbconvert \
  --to markdown \
  --output-dir "$output_dir" \
  examples/torchgeo_ftw_polygonize.ipynb

echo "Done. Run 'uv run --locked --extra docs zensical serve' to preview."
