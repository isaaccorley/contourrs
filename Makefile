.PHONY: install build test check clean

install:
	uv run pre-commit install
	uv run maturin develop --release

build:
	uv run maturin develop --release

test:
	cargo test --workspace
	uv run pytest tests/ -v

check:
	uv run pre-commit run --all-files

clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info .venv/
