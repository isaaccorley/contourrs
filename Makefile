.PHONY: install build test check docs clean

install:
	uv run --locked --extra dev pre-commit install
	uv run --locked --extra dev maturin develop --release

build:
	uv run --locked --extra dev maturin develop --release

test:
	cargo test --workspace --all-features
	uv run --locked --extra test pytest tests/ -v

check:
	uv run --locked --extra dev pre-commit run --all-files

docs:
	uv run --locked --extra docs zensical serve --dev-addr 0.0.0.0:8000

clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info .venv/
