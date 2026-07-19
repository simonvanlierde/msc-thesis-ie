# TODOs

- [ ] **Confirm the 2025 grid carbon intensity.** `[scenario."SQ"]` in
  `data/input/parameters/parameters.toml` is set to `carbon_intensity_electric_grid_kgCO2eq_kWh = 0.2622`
  (262.2 g CO₂/kWh, from electricitymaps.com). Verify this is the right figure for the SQ central
  year (2025) — check the exact metric (production vs. consumption / lifecycle vs. direct), the
  region (NL), and the averaging window — then confirm or correct the value.

## Follow-ups from the UHI-correction branch

- [ ] `dashboard/scripts/build_temporal.py` still calls the retired 3-arg `calc_Q_*` signatures
  (e.g. `calc_Q_transmission(batch, ts, gp)`) — will TypeError; needs updating to the
  `delta_T_air`-passing form.
- [ ] Pre-existing shapely `oriented_envelope` `RuntimeWarning`s show up in the GIS prep logs —
  investigate and silence.
- [ ] Add a `uhi_scale`-magnitude test: cooling demand at `uhi_scale=2` should exceed demand at
  `uhi_scale=1`.
- [ ] Make the `rasterio.transform` / `rasterio.windows` imports explicit (currently reached only
  through bare `import rasterio`) in `scripts/gis/add_uhi_to_buildings.py` and
  `tests/test_add_uhi_to_buildings.py`.
- [ ] Regenerate `docs/pipeline_dag.svg` — it predates the `fetch_uhi_habib` /
  `add_uhi_to_buildings` UHI-sampling split.
