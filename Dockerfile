# Hermetic runtime for the reproducible Snakemake pipeline.
# The uv.lock-pinned environment (GDAL/PROJ/GEOS via wheels + Python stack +
# Snakemake) is installed into /work/.venv, so the workflow runs without --sdm conda.
#
#   docker build -t cooling-demand .
#   docker run --rm -e EP_ONLINE_API_KEY -v "$PWD:/work" cooling-demand \
#     snakemake --cores 4
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /work

# Resolve dependencies before copying the source, so the layer caches across edits.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY . /work
RUN uv sync --locked

ENV PATH="/work/.venv/bin:$PATH"

CMD ["snakemake", "--cores", "1"]
