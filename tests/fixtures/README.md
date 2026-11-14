# Real BAG inputs with synthetic energy labels

`bag_2000.gpkg` contains 2,000 rows sampled from the prepared Hague building stock.
It has real BAG geometry, physical attributes and sampled Habib UHImax. It contains no EP-Online energy labels.
The integration test cycles independently through label indices 1 to 12 in BAG-ID order.
That hybrid test guards numerical changes; its total is not an estimate for the real stock.
The separate local real-stock test keeps its original reference and optional full input.

The sample includes in-use residential and office buildings, selected without reading any energy-label column.
Sort the source rows by BAG ID, sample with pandas `random_state=42`, then sort the 2,000 rows by BAG ID again.
The source is `results/geodata/BAG_buildings_with_residence_data_full.gpkg`, layer `BAG_buildings_full`.
The source can contain differing footprints for one BAG ID, so IDs can repeat in this fixture.
The source hash and fixture hash are recorded in `SHA256SUMS` and `provenance.json`.

Source credit and transformations: [the dataset card](../../data/README.md#collection-and-attribution).
The combined fixture inherits CC BY-SA 4.0 from Habib UHImax; see [licence text](../../LICENSES/CC-BY-SA-4.0.txt).
