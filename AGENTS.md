# AGENTS.md

## Tooling

Uses `uv` for Python, `cargo` for Rust, `pre-commit` for linting/formatting. A `Makefile` wraps common commands:

| Command | What it does |
|---------|-------------|
| `make install` | Install pre-commit hooks + build the extension (`maturin develop --release`) |
| `make build` | Rebuild the Python extension (Rust → `.so`) |
| `make test` | Run Rust tests (`cargo test`) + Python tests (`pytest`) |
| `make check` | Run all pre-commit hooks (ruff, formatting, etc.) |
| `make clean` | Remove build artifacts, cargo target, `.venv` |

### Typical workflow

```bash
make install   # first time setup
make build     # after Rust changes
make test      # run full test suite
make check     # lint/format gate (run before committing)
```

## Docs + notebook tutorials

- Docs are built with `zensical` from `docs/`.
- Real-world examples are notebook-first under `examples/`:
  - `examples/dem_contour.ipynb`
  - `examples/cdl_tiled_polygonize.ipynb`
  - `examples/torchgeo_ftw_polygonize.ipynb`
- During docs deploy (`.github/workflows/docs.yml`), notebooks are executed with `nbconvert` and rendered to Markdown in `docs/tutorials/` before `zensical build`.
- Cells tagged `skip_ci` are removed during CI execution/conversion to keep builds deterministic.
- Notebook CI (`.github/workflows/integration-notebooks.yml`) executes and converts tutorial notebooks on `push`, `pull_request`, and `workflow_dispatch`.

## Releasing

Tag-triggered releases. No auto-release on merge to main.

### Steps

1. Bump version in **both** `Cargo.toml` (workspace) and `pyproject.toml`
2. Commit: `chore: bump version to X.Y.Z`
3. Tag + push:
   ```bash
   git tag vX.Y.Z
   git push origin main --tags
   ```

### What happens

- GitHub Release created with auto-generated notes (`--generate-notes`) if one does not already exist for the tag
- Wheels built for linux x86/arm, macOS arm, Windows (py 3.12–3.14)
- sdist + wheels published to PyPI via trusted publishing
