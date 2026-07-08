lint: FORCE
	ruff check stitching/ experiments/ tests/
	ruff format --check stitching/ experiments/ tests/
	mypy stitching/ tests/

format:
	ruff check --fix stitching/ experiments/ tests/
	ruff format stitching/ experiments/ tests/

test: FORCE
	JAX_PLATFORMS=cpu pytest stitching/ tests/ -v -n auto -m "not slow"

# The full suite including the slow end-to-end parity gate (its reference CSV is
# platform-baked, so it is deselected from the default `make test`).
test-all: FORCE
	JAX_PLATFORMS=cpu pytest stitching/ tests/ -v -n auto

FORCE:
