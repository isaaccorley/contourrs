.PHONY: build dev test test-rust test-python lint fmt clippy check clean install-hooks hooks

# Build
build:
	uv run maturin develop --release

dev:
	uv run maturin develop

# Test
test: test-rust test-python

test-rust:
	cargo test --workspace

test-python: build
	uv run pytest tests/ -v

# Lint / Format
lint: fmt clippy check

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --features contourrs-core/arrow -- -D warnings

check:
	cargo check --workspace --features contourrs-core/arrow

# Pre-commit
install-hooks:
	uv run pre-commit install

hooks:
	uv run pre-commit run --all-files

# Clean
clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info .venv/
