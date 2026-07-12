# Hermetic runtime for the reproducible Snakemake pipeline.
# The uv.lock-pinned environment (GDAL/PROJ/GEOS via wheels + Python stack +
# Snakemake) is installed into /opt/venv, so the workflow runs without --sdm conda.
# The venv lives outside /work so that bind-mounting the repo does not shadow it.
#
#   docker build -t cooling-demand .
#   docker run --rm -e EP_ONLINE_API_KEY -v "$PWD:/work" cooling-demand \
#     snakemake --cores 4
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /work

# Resolve dependencies before the source is present, so the layer caches across edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --locked --no-dev --no-install-project

COPY . /work
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/opt/venv/bin:$PATH"

CMD ["snakemake", "--cores", "1"]
