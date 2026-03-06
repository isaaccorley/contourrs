.PHONY: install build test check docs clean

install:
	uv run pre-commit install
	uv run maturin develop --release

build:
	uv run maturin develop --release

test:
	cargo test --workspace
	uv run --extra dev pytest tests/ -v

check:
	uv run pre-commit run --all-files

docs:
	uv run --with mkdocs --with mkdocs-material mkdocs serve --dev-addr 0.0.0.0:8000

clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info .venv/
