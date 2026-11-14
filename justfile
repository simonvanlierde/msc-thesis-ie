# Run the same checks locally as in CI.
default:
    @just --list

check:
    uv run --locked ruff check .
    uv run --locked ruff format --check .
    uv run --locked ty check
    uv run --locked --group dataset pytest
    pnpm -C dashboard check

# Type-check the model and scripts.
types:
    uv run --locked ty check

fmt:
    uv run ruff check --fix .
    uv run ruff format .
    pnpm -C dashboard format

hooks:
    uv run prek run --all-files

licenses:
    uv run --locked python scripts/check_dependency_licenses.py
    requirements=$(mktemp) && trap 'rm -f "$requirements"' EXIT && uv export --locked --no-default-groups --no-hashes --no-annotate --no-header --no-emit-project -q -o "$requirements" && uv run --locked --with licensecheck==2026.0.8 licensecheck -r "$requirements" --zero
    pnpm -C dashboard run licenses
