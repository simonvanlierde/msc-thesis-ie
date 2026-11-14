import tempfile
from pathlib import Path

import yaml
from snakemake.utils import update_config

from cdm.constants import SCENARIOS
from scripts.gis.fetch_uhi_habib import download_file, ensure_rd_new, extract_member


def city_config(slug):
    with open("config/sources.yaml") as stream:
        defaults = yaml.safe_load(stream)
    with open(f"config/cities/{slug}.yaml") as stream:
        update_config(defaults, yaml.safe_load(stream))
    return defaults


CITY_CONFIGS = {slug: city_config(slug) for slug in ("oegstgeest", "voorschoten")}
HABIB = CITY_CONFIGS["oegstgeest"]["habib_uhi"]
HABIB_ARCHIVE = "data/raw/habib_uhi/UHI_NL.zip"

wildcard_constraints:
    city="oegstgeest|voorschoten"

localrules: all


rule all:
    input:
        expand("results/two-cities/{city}/CDM_results_{scenario}_full.csv",
               city=["oegstgeest", "voorschoten"], scenario=SCENARIOS),
        expand("results/two-cities/{city}/figures/scenario_overview.png",
               city=["oegstgeest", "voorschoten"]),


module oegstgeest:
    snakefile: "../Snakefile"
    config: CITY_CONFIGS["oegstgeest"]


module voorschoten:
    snakefile: "../Snakefile"
    config: CITY_CONFIGS["voorschoten"]


# Both city DAGs consume the same private national labels through one provider.
use rule provide_ep_online_energy_labels from oegstgeest

# Notebook and paper targets embed Hague-specific paths or interpretation. Keep them out
# of this prototype instead of presenting those outputs as validated for another city.
use rule fetch_city_boundary, fetch_bag_residences, discover_pdok_3d_height_tiles, download_pdok_3d_height_tiles, fetch_weather, prepare_bag_geodata, add_uhi_to_buildings, subsample_buildings, thermodynamic_model, lca, scenario_overview_figure from oegstgeest as oegstgeest_*
use rule fetch_city_boundary, fetch_bag_residences, discover_pdok_3d_height_tiles, download_pdok_3d_height_tiles, fetch_weather, prepare_bag_geodata, add_uhi_to_buildings, subsample_buildings, thermodynamic_model, lca, scenario_overview_figure from voorschoten as voorschoten_*


# One producer owns the mutable archive cache; extraction jobs only read that ZIP.
rule fetch_habib_archive:
    retries: 3
    input:
        environment="uv.lock",
        script="scripts/gis/fetch_uhi_habib.py",
        http="scripts/gis/pdok_http.py",
    output:
        HABIB_ARCHIVE,
    run:
        download_file(HABIB["files_api"], HABIB["archive_file_name"], Path(output[0]))


rule extract_habib_member:
    input:
        archive=HABIB_ARCHIVE,
        environment="uv.lock",
        script="scripts/gis/fetch_uhi_habib.py",
        http="scripts/gis/pdok_http.py",
    output:
        "data/raw/two-cities/{city}/uhi/habib.tif",
    params:
        member=lambda wildcards: CITY_CONFIGS[wildcards.city]["habib_uhi"]["uhimax_zip_member"],
    run:
        with tempfile.TemporaryDirectory() as directory:
            extracted = Path(directory) / "source.tif"
            extract_member(Path(input.archive), params.member, extracted)
            ensure_rd_new(extracted, Path(output[0]))
