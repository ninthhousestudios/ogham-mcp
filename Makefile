.PHONY: test lint build publish clean wheel

# Run all tests
test:
	uv run pytest tests/ -v

# Lint + format
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info

# Build wheel + sdist (clean first)
build: clean
	uv build

# Publish to PyPI (pulls token from 1Password)
publish: build
	uv publish --token $$(op read "op://Ogham-Gateway/PyPi - Ogham Dev token/api_key")

# Build wheel for gateway vendoring
wheel: build
	@echo "Wheel ready at dist/"
	@ls dist/*.whl
