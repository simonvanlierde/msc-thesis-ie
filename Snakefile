configfile: "config/sources.yaml"

SCENARIOS = ["SQ", "2030", "2050_L", "2050_M", "2050_H"]
SUBSET = "full"

PARAMETER_DIR = "data/input/parameters"
RAW_DIR = "data/raw"
RAW_GEODATA_DIR = f"{RAW_DIR}/geodata"

# All pipeline products go under RESULTS_DIR, leaving the committed reference
# outputs in data/output/ and docs/ untouched.
RESULTS_DIR = config.get("results_dir", "results")
RESULTS_GEODATA_DIR = f"{RESULTS_DIR}/geodata"
INTERMEDIATE_DIR = f"{RESULTS_DIR}/intermediate"

PDOK_BBOX = ",".join(str(value) for value in config["area"]["bbox"])
PDOK_BAG_BASE_URL = config["pdok_bag"]["base_url"]
PDOK_3D_BASE_URL = config["pdok_3d_basisvoorziening"]["base_url"]
PDOK_3D_YEAR = int(config["pdok_3d_basisvoorziening"]["year"])
EP_ONLINE_LABELS = config["ep_online"]["energy_labels_csv"]

# The stage scripts import the top-level ``functions`` package; running them by
# path puts scripts/ on sys.path instead of the repo root, so make the repo root
# importable for every rule.
shell.prefix("export PYTHONPATH=$(pwd):${{PYTHONPATH:-}}; ")

SCENARIO_PARAMETER_FILES = [
    f"{PARAMETER_DIR}/parameters_{{scenario}}/parameters_global.csv",
    f"{PARAMETER_DIR}/parameters_{{scenario}}/parameters_building_type.csv",
    f"{PARAMETER_DIR}/parameters_{{scenario}}/parameters_energy_class.csv",
    f"{PARAMETER_DIR}/parameters_{{scenario}}/parameters_cooling_technology.csv",
]


rule all:
    input:
        expand(f"{RESULTS_DIR}/CDM_results_{{scenario}}_" + SUBSET + ".csv", scenario=SCENARIOS),
        f"{RESULTS_DIR}/cooling_mix_elasticities_table.csv",
        f"{RESULTS_DIR}/figures/scenario_overview.png",


rule fetch_bag_residences:
    output:
        f"{RAW_DIR}/pdok_bag/verblijfsobject_the_hague.geojson",
    params:
        base_url=PDOK_BAG_BASE_URL,
        collection=config["pdok_bag"]["collections"]["residences"],
        bbox=PDOK_BBOX,
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        """
        python scripts/gis/fetch_ogc_features.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --bbox {params.bbox} \
          --output {output}
        """


rule discover_pdok_3d_height_tiles:
    output:
        f"{RAW_DIR}/pdok_3d_basisvoorziening/{PDOK_3D_YEAR}/height_tiles_manifest.json",
    params:
        base_url=PDOK_3D_BASE_URL,
        collection=config["pdok_3d_basisvoorziening"]["height_collection"],
        bbox=PDOK_BBOX,
        year=PDOK_3D_YEAR,
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        """
        python scripts/gis/discover_pdok_3d_height_tiles.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --bbox {params.bbox} \
          --year {params.year} \
          --output {output}
        """


rule download_pdok_3d_height_tiles:
    input:
        f"{RAW_DIR}/pdok_3d_basisvoorziening/{PDOK_3D_YEAR}/height_tiles_manifest.json",
    output:
        tiles=directory(f"{RAW_DIR}/pdok_3d_basisvoorziening/{PDOK_3D_YEAR}/height_tiles"),
        manifest=f"{RAW_DIR}/pdok_3d_basisvoorziening/{PDOK_3D_YEAR}/height_tiles_local_manifest.json",
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        """
        python scripts/gis/download_manifest_files.py \
          --manifest {input} \
          --output-dir {output.tiles} \
          --local-manifest {output.manifest}
        """


rule provide_ep_online_energy_labels:
    output:
        EP_ONLINE_LABELS,
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        "python scripts/gis/ensure_energy_labels.py --labels {output}"


rule prepare_bag_geodata:
    input:
        height_manifest=f"{RAW_DIR}/pdok_3d_basisvoorziening/{PDOK_3D_YEAR}/height_tiles_local_manifest.json",
        residences=f"{RAW_DIR}/pdok_bag/verblijfsobject_the_hague.geojson",
        energy_labels=EP_ONLINE_LABELS,
        script="scripts/gis/prepare_pdok_model_geodata.py",
    output:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
    params:
        layer=f"BAG_buildings_{SUBSET}",
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        """
        python scripts/gis/prepare_pdok_model_geodata.py \
          --height-manifest {input.height_manifest} \
          --bag-residences {input.residences} \
          --energy-labels {input.energy_labels} \
          --output {output.buildings} \
          --layer {params.layer}
        """


rule thermodynamic_model:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
        parameters=SCENARIO_PARAMETER_FILES,
        solar_fractions=f"{PARAMETER_DIR}/multidirectional_solar_radiation_fractions.csv",
        presence_load_factors=f"{PARAMETER_DIR}/presence_load_factors.csv",
        weather_backup=f"{PARAMETER_DIR}/raw_weather_data_2018_2022_HvH.csv",
    output:
        cooling_demand=f"{INTERMEDIATE_DIR}/buildings_with_cooling_demand_{{scenario}}_{SUBSET}.gpkg",
    params:
        buildings_layer=f"BAG_buildings_{SUBSET}",
        output_layer=lambda wildcards: f"buildings_with_cooling_demand_{wildcards.scenario}_{SUBSET}",
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        """
        python scripts/run_cdm_stage.py cooling-demand \
          --scenario {wildcards.scenario} \
          --buildings {input.buildings} \
          --buildings-layer {params.buildings_layer} \
          --solar-fractions {input.solar_fractions} \
          --presence-load-factors {input.presence_load_factors} \
          --output {output.cooling_demand} \
          --output-layer {params.output_layer}
        """


rule lca:
    input:
        cooling_demand=f"{INTERMEDIATE_DIR}/buildings_with_cooling_demand_{{scenario}}_{SUBSET}.gpkg",
        parameters=f"{PARAMETER_DIR}/parameters_{{scenario}}/parameters_global.csv",
    output:
        geodata=f"{RESULTS_GEODATA_DIR}/buildings_with_CDM_results_{{scenario}}_{SUBSET}.gpkg",
        csv=f"{RESULTS_DIR}/CDM_results_{{scenario}}_" + SUBSET + ".csv",
    params:
        cooling_demand_layer=lambda wildcards: f"buildings_with_cooling_demand_{wildcards.scenario}_{SUBSET}",
        geodata_output_layer=lambda wildcards: f"buildings_with_CDM_results_{wildcards.scenario}_{SUBSET}",
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        """
        python scripts/run_cdm_stage.py lca \
          --scenario {wildcards.scenario} \
          --cooling-demand {input.cooling_demand} \
          --cooling-demand-layer {params.cooling_demand_layer} \
          --geodata-output {output.geodata} \
          --geodata-output-layer {params.geodata_output_layer} \
          --csv-output {output.csv}
        """


rule scenario_overview_figure:
    input:
        expand(f"{RESULTS_DIR}/CDM_results_{{scenario}}_" + SUBSET + ".csv", scenario=SCENARIOS),
        script="docs/make_overview_figure.py",
    output:
        f"{RESULTS_DIR}/figures/scenario_overview.png",
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        "python docs/make_overview_figure.py --input-dir {RESULTS_DIR} --output {output}"


rule cooling_mix_sensitivity:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
        parameters=[f"{PARAMETER_DIR}/parameters_SQ/parameters_{group}.csv" for group in ("global", "building_type", "energy_class", "cooling_technology")],
        solar_fractions=f"{PARAMETER_DIR}/multidirectional_solar_radiation_fractions.csv",
        presence_load_factors=f"{PARAMETER_DIR}/presence_load_factors.csv",
        script="scripts/run_cooling_mix_sensitivity.py",
    output:
        f"{RESULTS_DIR}/cooling_mix_elasticities_table.csv",
    params:
        buildings_layer=f"BAG_buildings_{SUBSET}",
    conda:
        "workflow/envs/cooling-demand.yml",
    shell:
        """
        SA_IMAGE_DIR={RESULTS_DIR}/figures/SA \
        python scripts/run_cooling_mix_sensitivity.py \
          --scenario SQ \
          --buildings {input.buildings} \
          --buildings-layer {params.buildings_layer} \
          --solar-fractions {input.solar_fractions} \
          --presence-load-factors {input.presence_load_factors} \
          --output {output}
        """
