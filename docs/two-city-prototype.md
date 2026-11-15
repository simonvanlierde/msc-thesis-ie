# Run the model for two municipalities

The prototype runs Oegstgeest and Voorschoten through the existing acquisition, building preparation,
cooling-demand and LCA rules. Both municipalities have a UHImax member in the Habib national archive.
It uses one year of weather (2020, station 330) to keep the first run small.

These predictions need independent validation before use outside The Hague. The prototype does not
publish maps, change the shipped dashboard, execute the Hague-specific notebooks, or configure Slurm.

1. Install the locked environment with `uv sync --locked`.
2. Supply the private EP-Online export at `data/raw/ep_online/current/energy_labels.csv`, as described in
   [the pipeline guide](pipeline.md#ep-online-energy-labels).
3. Inspect both city DAGs without fetching data:

   ```bash
   uv run snakemake --snakefile workflow/two_cities.smk --cores 2 --dry-run
   ```

4. Run both cities:

   ```bash
   uv run snakemake --snakefile workflow/two_cities.smk --cores 2
   ```

The same workflow can run just one scenario in both municipalities:

```bash
uv run snakemake --snakefile workflow/two_cities.smk --cores 2 \
  results/two-cities/oegstgeest/CDM_results_SQ_full.csv \
  results/two-cities/voorschoten/CDM_results_SQ_full.csv
```

Each city has its own products, logs and benchmarks under `results/two-cities/<city>/`.
Fetched city inputs and raster outputs stay under `data/raw/two-cities/<city>/`.
The shared national label export has one provider rule, so the two city jobs cannot download it
concurrently. One producer downloads the shared Habib archive under `data/raw/habib_uhi/`; two extraction jobs
only read that ZIP and write separate city rasters. A cold run needs about 4.5 GB for the archive,
plus the building data and results. Do not put labels or LCA outputs in a public cache.

The entry point imports the existing Snakefile twice as Snakemake modules. The profiles in
`config/cities/` choose the municipality, exact Habib archive member, raster path, weather and output
roots. The original `uv run snakemake --cores 2` still runs The Hague at its original paths.
To run one profile independently, use `--configfile config/cities/oegstgeest.yaml` with the original
Snakefile. The two-city entry point deliberately uses the checked-in profiles; edit those profiles
rather than overriding their paths through `--config`.

The offline tests resolve both complete DAGs. They check separate
output roots, exact Habib members, one national label provider, and unchanged Hague paths.
A synthetic archive also checks that each extraction job writes the correct raster:

```bash
uv run pytest tests/test_two_cities.py
```

This establishes that two municipalities can use the same model pipeline with isolated products.
A general `{city}` workflow, a scheduler profile and broader validation remain separate work.
