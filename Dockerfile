# Hermetic runtime for the reproducible Snakemake pipeline.
# The pinned conda-forge env (GDAL/PROJ/GEOS + Python stack + Snakemake) is
# installed into the base environment, so the workflow runs without --sdm conda.
#
#   docker build -t cooling-demand .
#   docker run --rm -e EP_ONLINE_API_KEY -v "$PWD:/work" cooling-demand \
#     snakemake --cores 4
FROM mambaorg/micromamba:2.0-noble

COPY --chown=$MAMBA_USER:$MAMBA_USER workflow/envs/cooling-demand.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

# Activate the base env for every subsequent command.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /work
COPY --chown=$MAMBA_USER:$MAMBA_USER . /work

# The stage scripts import the top-level ``functions`` package by path.
ENV PYTHONPATH=/work

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["snakemake", "--cores", "1"]
