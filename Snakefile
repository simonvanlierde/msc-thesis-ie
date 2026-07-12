import re
from pathlib import Path

from cdm.constants import SCENARIOS

configfile: "config/sources.yaml"

# Target-only rules: no shell command, so never dispatch them to a cluster/executor.
localrules: all, cooling_mix, notebooks

SUBSET = config.get("building_subset", "full")
# main.ipynb's figures run on a representative sample of this many buildings (the full stock's
# hourly series do not fit in memory); only used when building_subset is "sample".
SAMPLE_SIZE = int(config.get("sample_size", 5000))
SAMPLE_SEED = int(config.get("sample_seed", 0))

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
BAG_RESIDENCES = f"{RAW_DIR}/pdok_bag/verblijfsobject_{CITY_SLUG}.gpkg"
WEATHER_CSV = f"{RESULTS_DIR}/weather/knmi_{WEATHER_STATION}_{WEATHER_START}_{WEATHER_END}.csv"

PDOK_3D_BASE_URL = config["pdok_3d_basisvoorziening"]["base_url"]
PDOK_3D_YEAR = int(config["pdok_3d_basisvoorziening"]["year"])
PDOK_3D_DIR = f"{RAW_DIR}/pdok_3d_basisvoorziening/{CITY_SLUG}/{PDOK_3D_YEAR}"
EP_ONLINE_LABELS = config["ep_online"]["energy_labels_csv"]

CBS_POSTCODE4 = f"{RAW_DIR}/cbs/postcode4_{CITY_SLUG}.gpkg"
CBS_BUURTEN = f"{RAW_DIR}/cbs/buurten_{CITY_SLUG}.gpkg"

RIVM_UHI_ZIP_URL = config["rivm_uhi"]["zip_url"]
RIVM_UHI_CACHE_DIR = f"{RAW_DIR}/rivm_uhi"
# gis.ipynb reads this exact path (a clip of the national raster). The rule
# materialises it, so the notebook's one external raster becomes reproducible.
UHI_RASTER = "data/input/geodata/UHI_effect_TheHague.tif"

# The model rules call ``cdm`` through a shell command, so Snakemake's ``code``
# rerun-trigger (which only hashes run:/script: bodies) cannot see it. Declare the
# model sources as inputs so editing the model re-runs the rules that use it.
MODEL_SRC = sorted(str(path) for path in Path("cdm").glob("*.py"))

# Rules that hit PDOK/KNMI/EP-Online. A 5xx mid-pagination kills the whole script, which
# pdok_http's in-request Retry adapter cannot resume; re-running the job is the only recovery.
NETWORK_RETRIES = 3


rule all:
    input:
        expand(f"{RESULTS_DIR}/CDM_results_{{scenario}}_{SUBSET}.csv", scenario=SCENARIOS),
        f"{RESULTS_DIR}/figures/scenario_overview.png",


# Heavy opt-in target (30 tech pairs x 20 model runs); not part of `all`.
rule cooling_mix:
    input:
        f"{RESULTS_DIR}/cooling_mix_elasticities_table.csv",


# Executed-notebook target: run both analysis notebooks headless and keep the
# executed copies (figures, maps and sensitivity analyses embedded) as artifacts.
# Opt-in and never part of `all` — main.ipynb keeps hourly series per building
# and the curated sensitivity sweep runs the model dozens of times, so this is
# deliberately heavy.
rule notebooks:
    input:
        f"{RESULTS_DIR}/notebooks/main.executed.ipynb",
        f"{RESULTS_DIR}/notebooks/gis.executed.ipynb",
        f"{RESULTS_DIR}/sensitivity_summary.csv",


rule fetch_city_boundary:
    retries: NETWORK_RETRIES
    output:
        boundary=BOUNDARY_GEOJSON,
        bbox=BBOX_FILE,
    params:
        base_url=config["pdok_boundary"]["base_url"],
        collection=config["pdok_boundary"]["collection"],
        name=CITY_NAME,
    log:
        f"{LOG_DIR}/fetch_city_boundary.log",
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
    retries: NETWORK_RETRIES
    input:
        bbox=BBOX_FILE,
    output:
        BAG_RESIDENCES,
    params:
        base_url=config["pdok_bag"]["base_url"],
        collection=config["pdok_bag"]["collections"]["residences"],
    log:
        f"{LOG_DIR}/fetch_bag_residences.log",
    shell:
        """
        python scripts/gis/fetch_ogc_features.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --bbox $(cat {input.bbox}) \
          --output {output} > {log} 2>&1
        """


rule discover_pdok_3d_height_tiles:
    retries: NETWORK_RETRIES
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
    retries: NETWORK_RETRIES
    input:
        f"{PDOK_3D_DIR}/height_tiles_manifest.json",
    output:
        tiles=directory(f"{PDOK_3D_DIR}/height_tiles"),
        manifest=f"{PDOK_3D_DIR}/height_tiles_local_manifest.json",
    log:
        f"{LOG_DIR}/download_pdok_3d_height_tiles.log",
    shell:
        """
        python scripts/gis/download_manifest_files.py \
          --manifest {input} \
          --output-dir {output.tiles} \
          --local-manifest {output.manifest} > {log} 2>&1
        """


# CBS geographic divisions for gis.ipynb. Together with the city boundary (fetch_city_boundary)
# these replace the committed GeographicDivisions GeoPackage, so gis has no un-reproducible input.
rule fetch_cbs_postcode4:
    retries: NETWORK_RETRIES
    input:
        bbox=BBOX_FILE,
    output:
        CBS_POSTCODE4,
    params:
        base_url=config["cbs_postcode4"]["base_url"],
        collection=config["cbs_postcode4"]["collection"],
        # The postcode polygons are detailed and this collection is slow to serve: page small and
        # allow a long read timeout, or the request dies before the response completes.
        limit=200,
        timeout=300,
    log:
        f"{LOG_DIR}/fetch_cbs_postcode4.log",
    shell:
        """
        python scripts/gis/fetch_ogc_features.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --bbox $(cat {input.bbox}) \
          --limit {params.limit} \
          --timeout {params.timeout} \
          --output {output} > {log} 2>&1
        """


rule fetch_cbs_buurten:
    retries: NETWORK_RETRIES
    input:
        bbox=BBOX_FILE,
    output:
        CBS_BUURTEN,
    params:
        base_url=config["cbs_wijken_en_buurten"]["base_url"],
        collection=config["cbs_wijken_en_buurten"]["collection"],
    log:
        f"{LOG_DIR}/fetch_cbs_buurten.log",
    shell:
        """
        python scripts/gis/fetch_ogc_features.py \
          --base-url {params.base_url} \
          --collection {params.collection} \
          --bbox $(cat {input.bbox}) \
          --output {output} > {log} 2>&1
        """


rule fetch_uhi_raster:
    # Heavy: downloads a ~1.95 GB national raster (cached), then clips to the city bbox.
    # Only needed by run_gis_notebook, never by `all`.
    retries: NETWORK_RETRIES
    input:
        bbox=BBOX_FILE,
    output:
        UHI_RASTER,
    params:
        url=RIVM_UHI_ZIP_URL,
        cache_dir=RIVM_UHI_CACHE_DIR,
    log:
        f"{LOG_DIR}/fetch_uhi_raster.log",
    shell:
        """
        python scripts/gis/fetch_uhi_raster.py \
          --url {params.url} \
          --bbox $(cat {input.bbox}) \
          --cache-dir {params.cache_dir} \
          --output {output} > {log} 2>&1
        """


rule provide_ep_online_energy_labels:
    retries: NETWORK_RETRIES
    output:
        EP_ONLINE_LABELS,
    log:
        f"{LOG_DIR}/provide_ep_online_energy_labels.log",
    shell:
        "python scripts/gis/ensure_energy_labels.py --labels {output} > {log} 2>&1"


rule fetch_weather:
    retries: NETWORK_RETRIES
    # NOTE: no model_src input, even though fetch_weather.py imports
    # cdm.time_series. It only calls the KNMI downloader, and listing MODEL_SRC here
    # would re-download the series on every model edit. Add cdm/time_series.py alone
    # if get_raw_weather_data ever starts transforming the data.
    output:
        WEATHER_CSV,
    params:
        station=WEATHER_STATION,
        start_year=WEATHER_START,
        end_year=WEATHER_END,
    log:
        f"{LOG_DIR}/fetch_weather.log",
    shell:
        """
        python scripts/gis/fetch_weather.py \
          --station {params.station} \
          --start-year {params.start_year} \
          --end-year {params.end_year} \
          --output {output} > {log} 2>&1
        """


# Always builds the full stock. The optional `sample` subset is a downstream selection
# (subsample_buildings), so this output is pinned to `_full` regardless of SUBSET.
rule prepare_bag_geodata:
    input:
        height_manifest=f"{PDOK_3D_DIR}/height_tiles_local_manifest.json",
        residences=BAG_RESIDENCES,
        energy_labels=EP_ONLINE_LABELS,
        boundary=BOUNDARY_GEOJSON,
        script="scripts/gis/prepare_pdok_model_geodata.py",
        model_src=MODEL_SRC,
    output:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_full.gpkg",
    params:
        layer="BAG_buildings_full",
    log:
        f"{LOG_DIR}/prepare_bag_geodata.log",
    benchmark:
        f"{BENCHMARK_DIR}/prepare_bag_geodata.tsv"
    threads: 1
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


# Representative subset for the memory-bound notebook figures. Reads the full stock and
# writes a schema-identical GeoPackage, so every downstream rule treats `sample` like `full`.
rule subsample_buildings:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_full.gpkg",
        script="scripts/subsample_buildings.py",
        model_src=MODEL_SRC,
    output:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_sample.gpkg",
    params:
        input_layer="BAG_buildings_full",
        output_layer="BAG_buildings_sample",
        sample_size=SAMPLE_SIZE,
        seed=SAMPLE_SEED,
    log:
        f"{LOG_DIR}/subsample_buildings.log",
    shell:
        """
        python scripts/subsample_buildings.py \
          --buildings {input.buildings} \
          --buildings-layer {params.input_layer} \
          --output {output.buildings} \
          --output-layer {params.output_layer} \
          --sample-size {params.sample_size} \
          --seed {params.seed} > {log} 2>&1
        """


rule thermodynamic_model:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
        parameters=[PARAMETERS_TOML, *PARAMETER_GROUP_FILES],
        weather=WEATHER_CSV,
        solar_fractions=f"{PARAMETER_DIR}/multidirectional_solar_radiation_fractions.csv",
        presence_load_factors=f"{PARAMETER_DIR}/presence_load_factors.csv",
        script="scripts/run_cdm_stage.py",
        model_src=MODEL_SRC,
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
        script="scripts/run_cdm_stage.py",
        model_src=MODEL_SRC,
    output:
        geodata=f"{RESULTS_GEODATA_DIR}/buildings_with_CDM_results_{{scenario}}_{SUBSET}.gpkg",
        csv=f"{RESULTS_DIR}/CDM_results_{{scenario}}_{SUBSET}.csv",
    params:
        cooling_demand_layer=lambda wildcards: f"buildings_with_cooling_demand_{wildcards.scenario}_{SUBSET}",
        geodata_output_layer=lambda wildcards: f"buildings_with_CDM_results_{wildcards.scenario}_{SUBSET}",
    log:
        f"{LOG_DIR}/lca_{{scenario}}.log",
    benchmark:
        f"{BENCHMARK_DIR}/lca_{{scenario}}.tsv"
    threads: 1
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
        csvs=expand(f"{RESULTS_DIR}/CDM_results_{{scenario}}_{SUBSET}.csv", scenario=SCENARIOS),
        script="docs/make_overview_figure.py",
    output:
        f"{RESULTS_DIR}/figures/scenario_overview.png",
    log:
        f"{LOG_DIR}/scenario_overview_figure.log",
    shell:
        # Derive the input dir from a declared input rather than the global, so provenance stays accurate.
        "python docs/make_overview_figure.py --input-dir $(dirname {input.csvs[0]}) --output {output} > {log} 2>&1"


rule cooling_mix_sensitivity:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
        parameters=[PARAMETERS_TOML, *PARAMETER_GROUP_FILES],
        weather=WEATHER_CSV,
        solar_fractions=f"{PARAMETER_DIR}/multidirectional_solar_radiation_fractions.csv",
        presence_load_factors=f"{PARAMETER_DIR}/presence_load_factors.csv",
        script="scripts/run_cooling_mix_sensitivity.py",
        model_src=MODEL_SRC,
    output:
        table=f"{RESULTS_DIR}/cooling_mix_elasticities_table.csv",
        # The sweep writes two PNGs per technology pair; declare the directory so
        # Snakemake can clean and invalidate them instead of them appearing from nowhere.
        figures=directory(f"{RESULTS_DIR}/figures/SA"),
    params:
        buildings_layer=f"BAG_buildings_{SUBSET}",
    log:
        f"{LOG_DIR}/cooling_mix_sensitivity.log",
    benchmark:
        f"{BENCHMARK_DIR}/cooling_mix_sensitivity.tsv"
    threads: 1
    shell:
        """
        python scripts/run_cooling_mix_sensitivity.py \
          --scenario SQ \
          --buildings {input.buildings} \
          --buildings-layer {params.buildings_layer} \
          --solar-fractions {input.solar_fractions} \
          --presence-load-factors {input.presence_load_factors} \
          --weather-csv {input.weather} \
          --image-dir {output.figures} \
          --output {output.table} > {log} 2>&1
        """


# Curated sensitivity-analysis set (the 15 SAs the paper draft leans on), downsampled onto the
# sample subset. Replaces the ~20-SA main.ipynb section. Opt-in; pulled by `notebooks`.
rule sensitivity:
    input:
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_{SUBSET}.gpkg",
        parameters=[PARAMETERS_TOML, *PARAMETER_GROUP_FILES],
        weather=WEATHER_CSV,
        solar_fractions=f"{PARAMETER_DIR}/multidirectional_solar_radiation_fractions.csv",
        presence_load_factors=f"{PARAMETER_DIR}/presence_load_factors.csv",
        script="scripts/run_sensitivity.py",
        model_src=MODEL_SRC,
    output:
        summary=f"{RESULTS_DIR}/sensitivity_summary.csv",
        figures=directory(f"{RESULTS_DIR}/figures/SA_curated"),
    params:
        buildings_layer=f"BAG_buildings_{SUBSET}",
        steps=int(config.get("sa_steps", 21)),
    log:
        f"{LOG_DIR}/sensitivity.log",
    benchmark:
        f"{BENCHMARK_DIR}/sensitivity.tsv"
    threads: 1
    shell:
        """
        python scripts/run_sensitivity.py \
          --scenario SQ \
          --buildings {input.buildings} \
          --buildings-layer {params.buildings_layer} \
          --solar-fractions {input.solar_fractions} \
          --presence-load-factors {input.presence_load_factors} \
          --weather-csv {input.weather} \
          --image-dir {output.figures} \
          --steps {params.steps} \
          --output {output.summary} > {log} 2>&1
        """


rule run_main_notebook:
    input:
        # Pinned to the sample subset regardless of SUBSET: the notebook keeps hourly series for
        # every building it models, which the full stock cannot fit in memory (~90+ GB).
        buildings=f"{RESULTS_GEODATA_DIR}/BAG_buildings_with_residence_data_sample.gpkg",
        parameters=[PARAMETERS_TOML, *PARAMETER_GROUP_FILES],
        weather=WEATHER_CSV,
        notebook="main.ipynb",
        model_src=MODEL_SRC,
    output:
        notebook=f"{RESULTS_DIR}/notebooks/main.executed.ipynb",
        # The notebook's figures now write here (repointed off the old data/output/ fork), so
        # declare the directory to make them tracked, cleanable pipeline artifacts.
        figures=directory(f"{RESULTS_DIR}/figures/main"),
    log:
        f"{LOG_DIR}/run_main_notebook.log",
    shell:
        # timeout=-1: the scenario re-run legitimately takes a long time headless.
        # BUILDING_SUBSET_NAME: the notebook reads it (cell 5) to pick the buildings layer.
        # The notebook reads the fetched weather CSV itself (falling back to a live KNMI fetch
        # only when it is absent), so the declared weather input keeps it hermetic.
        """
        BUILDING_SUBSET_NAME=sample \
        jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 \
          --output-dir "$(dirname {output.notebook})" --output "$(basename {output.notebook})" \
          {input.notebook} > {log} 2>&1
        """


# The dashboard's precomputed JSON is committed (the site deploys without a pipeline run), so
# this is opt-in and not part of `all`: `snakemake dashboard_data` regenerates it in place and
# reports it as out-of-date whenever the model, parameters or results move underneath it.
rule dashboard_data:
    input:
        # Pinned to the full stock (not SUBSET), like run_gis_notebook: the dashboard ships the
        # whole city, and the build scripts resolve `*_full` filenames themselves.
        cdm_csvs=expand(f"{RESULTS_DIR}/CDM_results_{{scenario}}_full.csv", scenario=SCENARIOS),
        cdm_geodata=expand(
            f"{RESULTS_GEODATA_DIR}/buildings_with_CDM_results_{{scenario}}_full.gpkg",
            scenario=SCENARIOS,
        ),
        # build_choropleth aggregates the buildings to the same CBS buurten gis.ipynb uses, and
        # build_temporal re-runs the heat balance on a building sample, so it reads the parameters
        # and the model directly.
        buurten=CBS_BUURTEN,
        weather=WEATHER_CSV,
        parameters=[PARAMETERS_TOML, *PARAMETER_GROUP_FILES],
        scripts=[
            "dashboard/scripts/build_scenarios.py",
            "dashboard/scripts/build_choropleth.py",
            "dashboard/scripts/build_temporal.py",
        ],
        model_src=MODEL_SRC,
    output:
        scenarios="dashboard/public/data/scenarios.json",
        choropleth="dashboard/public/data/cooling_by_buurt.geojson",
        temporal="dashboard/public/data/temporal.json",
    params:
        city=CITY_NAME,
    log:
        f"{LOG_DIR}/dashboard_data.log",
    shell:
        """
        {{
          python dashboard/scripts/build_scenarios.py \
            --results-dir {RESULTS_DIR} --out {output.scenarios}
          python dashboard/scripts/build_choropleth.py \
            --divisions {input.buurten} --city "{params.city}" \
            --geodata-dir {RESULTS_GEODATA_DIR} --out {output.choropleth}
          python dashboard/scripts/build_temporal.py \
            --geodata-dir {RESULTS_GEODATA_DIR} --results-dir {RESULTS_DIR} \
            --weather-csv {input.weather} --out {output.temporal}
        }} > {log} 2>&1
        """


rule run_gis_notebook:
    input:
        # gis.ipynb pins SCENARIO="SQ". Every input it reads is now produced by a rule -- the UHI
        # raster, the CBS postcode areas (income) and buurten (neighbourhoods), and the municipal
        # boundary (the city outline). The committed GeographicDivisions GeoPackage is gone.
        # Pinned to the full stock (not SUBSET): the maps are spatial aggregations, not memory-bound
        # time series, so they want complete coverage. Only main.ipynb needs the sample subset.
        cdm_csv=f"{RESULTS_DIR}/CDM_results_SQ_full.csv",
        cdm_geodata=f"{RESULTS_GEODATA_DIR}/buildings_with_CDM_results_SQ_full.gpkg",
        uhi_raster=UHI_RASTER,
        postcode4=CBS_POSTCODE4,
        buurten=CBS_BUURTEN,
        boundary=BOUNDARY_GEOJSON,
        notebook="gis.ipynb",
        model_src=MODEL_SRC,
    output:
        notebook=f"{RESULTS_DIR}/notebooks/gis.executed.ipynb",
        figures=directory(f"{RESULTS_DIR}/figures/gis"),
    log:
        f"{LOG_DIR}/run_gis_notebook.log",
    shell:
        # BUILDING_SUBSET_NAME=full: the notebook reads it (cell 31) to pick the CDM layer, matching
        # the full-stock inputs declared above.
        """
        BUILDING_SUBSET_NAME=full \
        jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 \
          --output-dir "$(dirname {output.notebook})" --output "$(basename {output.notebook})" \
          {input.notebook} > {log} 2>&1
        """
