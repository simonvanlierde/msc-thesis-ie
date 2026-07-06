import re

configfile: "config/sources.yaml"

SCENARIOS = ["SQ", "2030", "2050_L", "2050_M", "2050_H"]
SUBSET = "full"

PARAMETER_DIR = "data/input/parameters"
PARAMETERS_TOML = f"{PARAMETER_DIR}/parameters.toml"
PARAMETER_GROUP_FILES = [
    f"{PARAMETER_DIR}/parameters_building_type.csv",
    f"{PARAMETER_DIR}/parameters_energy_class.csv",
    f"{PARAMETER_DIR}/parameters_cooling_technology.csv",
]
RAW_DIR = "data/raw"

# All pipeline products go under RESULTS_DIR, leaving the committed reference
# outputs in data/output/ and docs/ untouched.
RESULTS_DIR = config.get("results_dir", "results")
RESULTS_GEODATA_DIR = f"{RESULTS_DIR}/geodata"
INTERMEDIATE_DIR = f"{RESULTS_DIR}/intermediate"
LOG_DIR = f"{RESULTS_DIR}/logs"
BENCHMARK_DIR = f"{RESULTS_DIR}/benchmarks"

# Active city (name -> PDOK boundary -> bbox). CITY_SLUG namespaces the raw
# fetches so switching cities never reuses another city's data.
CITY_NAME = config["city"]["name"]
CITY_SLUG = re.sub(r"[^a-z0-9]+", "-", CITY_NAME.lower()).strip("-")
WEATHER_STATION = int(config["city"]["weather_station"])
WEATHER_START = int(config["weather"]["start_year"])
WEATHER_END = int(config["weather"]["end_year"])

BOUNDARY_GEOJSON = f"{RAW_DIR}/pdok_boundary/{CITY_SLUG}_boundary.geojson"
BBOX_FILE = f"{RAW_DIR}/pdok_boundary/{CITY_SLUG}_bbox.txt"
BAG_RESIDENCES = f"{RAW_DIR}/pdok_bag/verblijfsobject_{CITY_SLUG}.geojson"
WEATHER_CSV = f"{RESULTS_DIR}/weather/knmi_{WEATHER_STATION}_{WEATHER_START}_{WEATHER_END}.csv"

PDOK_3D_BASE_URL = config["pdok_3d_basisvoorziening"]["base_url"]
PDOK_3D_YEAR = int(config["pdok_3d_basisvoorziening"]["year"])
PDOK_3D_DIR = f"{RAW_DIR}/pdok_3d_basisvoorziening/{CITY_SLUG}/{PDOK_3D_YEAR}"
EP_ONLINE_LABELS = config["ep_online"]["energy_labels_csv"]

# The stage scripts import the top-level ``functions`` package; running them by
# path puts scripts/ on sys.path instead of the repo root, so make the repo root
# importable for every rule.
shell.prefix("export PYTHONPATH=$(pwd):${{PYTHONPATH:-}}; ")


rule all:
    input:
        expand(f"{RESULTS_DIR}/CDM_results_{{scenario}}_" + SUBSET + ".csv", scenario=SCENARIOS),
        f"{RESULTS_DIR}/figures/scenario_overview.png",


# Heavy opt-in target (30 tech pairs x 20 model runs); not part of `all`.
rule cooling_mix:
    input:
        f"{RESULTS_DIR}/cooling_mix_elasticities_table.csv",


rule fetch_city_boundary:
    output:
        boundary=BOUNDARY_GEOJSON,
        bbox=BBOX_FILE,
    params:
        base_url=config["pdok_boundary"]["base_url"],
        collection=config["pdok_boundary"]["collection"],
        name=CITY_NAME,
    log:
        f"{LOG_DIR}/fetch_city_boundary.log",
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/gis/fetch_city_boundary.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --name "{params.name}" \
          --output {output.boundary} \
          --bbox-output {output.bbox} > {log} 2>&1
        """


rule fetch_bag_residences:
    input:
        bbox=BBOX_FILE,
    output:
        BAG_RESIDENCES,
    params:
        base_url=config["pdok_bag"]["base_url"],
        collection=config["pdok_bag"]["collections"]["residences"],
    log:
        f"{LOG_DIR}/fetch_bag_residences.log",
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/gis/fetch_ogc_features.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --bbox $(cat {input.bbox}) \
          --output {output} > {log} 2>&1
        """


rule discover_pdok_3d_height_tiles:
    input:
        bbox=BBOX_FILE,
    output:
        f"{PDOK_3D_DIR}/height_tiles_manifest.json",
    params:
        base_url=PDOK_3D_BASE_URL,
        collection=config["pdok_3d_basisvoorziening"]["height_collection"],
        year=PDOK_3D_YEAR,
    log:
        f"{LOG_DIR}/discover_pdok_3d_height_tiles.log",
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/gis/discover_pdok_3d_height_tiles.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --bbox $(cat {input.bbox}) \
          --year {params.year} \
          --output {output} > {log} 2>&1
        """


rule download_pdok_3d_height_tiles:
    input:
        f"{PDOK_3D_DIR}/height_tiles_manifest.json",
    output:
        tiles=directory(f"{PDOK_3D_DIR}/height_tiles"),
        manifest=f"{PDOK_3D_DIR}/height_tiles_local_manifest.json",
    log:
        f"{LOG_DIR}/download_pdok_3d_height_tiles.log",
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/gis/download_manifest_files.py \
          --manifest {input} \
          --output-dir {output.tiles} \
          --local-manifest {output.manifest} > {log} 2>&1
        """


rule provide_ep_online_energy_labels:
    output:
        EP_ONLINE_LABELS,
    log:
        f"{LOG_DIR}/provide_ep_online_energy_labels.log",
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        "python scripts/gis/ensure_energy_labels.py --labels {output} > {log} 2>&1"


rule fetch_weather:
    output:
        WEATHER_CSV,
    params:
        station=WEATHER_STATION,
        start_year=WEATHER_START,
        end_year=WEATHER_END,
    log:
        f"{LOG_DIR}/fetch_weather.log",
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/gis/fetch_weather.py \
          --station {params.station} \
          --start-year {params.start_year} \
          --end-year {params.end_year} \
          --output {output} > {log} 2>&1
        """


rule prepare_bag_geodata:
    input:
        height_manifest=f"{PDOK_3D_DIR}/height_tiles_local_manifest.json",
        residences=BAG_RESIDENCES,
        energy_labels=EP_ONLINE_LABELS,
        boundary=BOUNDARY_GEOJSON,
        script="scripts/gis/prepare_pdok_model_geodata.py",
    output:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
    params:
        layer=f"BAG_buildings_{SUBSET}",
    log:
        f"{LOG_DIR}/prepare_bag_geodata.log",
    benchmark:
        f"{BENCHMARK_DIR}/prepare_bag_geodata.tsv"
    threads: 1
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/gis/prepare_pdok_model_geodata.py \
          --height-manifest {input.height_manifest} \
          --bag-residences {input.residences} \
          --energy-labels {input.energy_labels} \
          --boundary {input.boundary} \
          --output {output.buildings} \
          --layer {params.layer} > {log} 2>&1
        """


rule thermodynamic_model:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
        parameters=[PARAMETERS_TOML, *PARAMETER_GROUP_FILES],
        weather=WEATHER_CSV,
        solar_fractions=f"{PARAMETER_DIR}/multidirectional_solar_radiation_fractions.csv",
        presence_load_factors=f"{PARAMETER_DIR}/presence_load_factors.csv",
    output:
        cooling_demand=f"{INTERMEDIATE_DIR}/buildings_with_cooling_demand_{{scenario}}_{SUBSET}.gpkg",
    params:
        buildings_layer=f"BAG_buildings_{SUBSET}",
        output_layer=lambda wildcards: f"buildings_with_cooling_demand_{wildcards.scenario}_{SUBSET}",
    log:
        f"{LOG_DIR}/thermodynamic_model_{{scenario}}.log",
    benchmark:
        f"{BENCHMARK_DIR}/thermodynamic_model_{{scenario}}.tsv"
    threads: 1
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/run_cdm_stage.py cooling-demand \
          --scenario {wildcards.scenario} \
          --buildings {input.buildings} \
          --buildings-layer {params.buildings_layer} \
          --solar-fractions {input.solar_fractions} \
          --presence-load-factors {input.presence_load_factors} \
          --weather-csv {input.weather} \
          --output {output.cooling_demand} \
          --output-layer {params.output_layer} > {log} 2>&1
        """


rule lca:
    input:
        cooling_demand=f"{INTERMEDIATE_DIR}/buildings_with_cooling_demand_{{scenario}}_{SUBSET}.gpkg",
        parameters=PARAMETERS_TOML,
    output:
        geodata=f"{RESULTS_GEODATA_DIR}/buildings_with_CDM_results_{{scenario}}_{SUBSET}.gpkg",
        csv=f"{RESULTS_DIR}/CDM_results_{{scenario}}_" + SUBSET + ".csv",
    params:
        cooling_demand_layer=lambda wildcards: f"buildings_with_cooling_demand_{wildcards.scenario}_{SUBSET}",
        geodata_output_layer=lambda wildcards: f"buildings_with_CDM_results_{wildcards.scenario}_{SUBSET}",
    log:
        f"{LOG_DIR}/lca_{{scenario}}.log",
    benchmark:
        f"{BENCHMARK_DIR}/lca_{{scenario}}.tsv"
    threads: 1
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        python scripts/run_cdm_stage.py lca \
          --scenario {wildcards.scenario} \
          --cooling-demand {input.cooling_demand} \
          --cooling-demand-layer {params.cooling_demand_layer} \
          --geodata-output {output.geodata} \
          --geodata-output-layer {params.geodata_output_layer} \
          --csv-output {output.csv} > {log} 2>&1
        """


rule scenario_overview_figure:
    input:
        expand(f"{RESULTS_DIR}/CDM_results_{{scenario}}_" + SUBSET + ".csv", scenario=SCENARIOS),
        script="docs/make_overview_figure.py",
    output:
        f"{RESULTS_DIR}/figures/scenario_overview.png",
    log:
        f"{LOG_DIR}/scenario_overview_figure.log",
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        "python docs/make_overview_figure.py --input-dir {RESULTS_DIR} --output {output} > {log} 2>&1"


rule cooling_mix_sensitivity:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
        parameters=[PARAMETERS_TOML, *PARAMETER_GROUP_FILES],
        weather=WEATHER_CSV,
        solar_fractions=f"{PARAMETER_DIR}/multidirectional_solar_radiation_fractions.csv",
        presence_load_factors=f"{PARAMETER_DIR}/presence_load_factors.csv",
        script="scripts/run_cooling_mix_sensitivity.py",
    output:
        f"{RESULTS_DIR}/cooling_mix_elasticities_table.csv",
    params:
        buildings_layer=f"BAG_buildings_{SUBSET}",
    log:
        f"{LOG_DIR}/cooling_mix_sensitivity.log",
    benchmark:
        f"{BENCHMARK_DIR}/cooling_mix_sensitivity.tsv"
    threads: 1
    conda:
        "workflow/envs/cooling-demand.yml"
    shell:
        """
        SA_IMAGE_DIR={RESULTS_DIR}/figures/SA \
        python scripts/run_cooling_mix_sensitivity.py \
          --scenario SQ \
          --buildings {input.buildings} \
          --buildings-layer {params.buildings_layer} \
          --solar-fractions {input.solar_fractions} \
          --presence-load-factors {input.presence_load_factors} \
          --weather-csv {input.weather} \
          --output {output} > {log} 2>&1
        """
