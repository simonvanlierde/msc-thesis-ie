---
license: cc-by-sa-4.0
language: en
size_categories:
  - 100K<n<1M
tags:
  - urban-heat-island
  - building-stock
  - the-hague
---

# The Hague building inputs with local UHImax

This open subset contains building geometry and physical inputs with corrected local urban heat island (UHI) maxima.
It excludes energy labels and all environmental impact results. It cannot reproduce the published cooling totals on its own.

## Motivation and uses

Use these inputs to inspect the spatial UHI correction or prepare a building model with independently supplied energy labels.
The data describe building stock, not measured indoor temperatures or cooling equipment ownership.

## Composition

`buildings.gpkg`, `buildings.parquet` and `buildings.csv` contain the same rows and columns.
The GeoPackage layer is `buildings`. GeoParquet uses zstd compression.
The CSV stores polygons as WKT. All coordinates use Amersfoort / RD New (EPSG:28992), in metres.
`datapackage.json` describes the columns; `inventory.json` records row counts, duplicate BAG IDs and missing values.
Empty CSV cells and Parquet nulls mean missing data, never zero.
BAG IDs can repeat for differing source footprints. These rows preserve the source stock and must not be counted as distinct buildings.
No person names, household identities, addresses or energy-label registration identifiers are included.
Footprints and BAG IDs identify publicly mapped buildings.

## Collection and attribution

Sources and terms checked on 2026-11-14:

- Kadaster BAG, obtained through [PDOK BAG](https://api.pdok.nl/kadaster/bag/ogc/v2), Public Domain Mark 1.0.
  Building use, residence counts, construction years and floor areas come from this register.
- Kadaster 3D Basisvoorziening, 2025 height tiles, [PDOK source and CC BY 4.0 terms](https://api.pdok.nl/kadaster/3d-basisvoorziening/ogc/v1).
  Roof and ground elevations and footprints form the physical building inputs.
- Habib, M., van Esch, M., van Ham, M., and Timmermans, W. (2025).
  *High-Resolution Datasets for Heat Vulnerability Assessment in High-Density Urban Areas of the Netherlands*, version 1.
  [DOI: 10.4121/0159bc0e-3ab9-42b6-8b12-ad3b2294face.v1](https://doi.org/10.4121/0159bc0e-3ab9-42b6-8b12-ad3b2294face.v1), CC BY-SA 4.0.
  Share-alike applies to this combined subset; see `LICENSES/CC-BY-SA-4.0.txt`.

## Preprocessing and limitations

The pipeline clips footprints to The Hague and joins in-use BAG residences spatially.
It computes building heights, volumes, bounding rectangles, dominant use and total residence floor areas.
UHImax averages valid raster pixels in a square of radius 30 metres around each building's representative point.
A building without covered pixels receives the configured 1 K fallback. The present output does not distinguish fallback values.
The source archive version and preprocessing code determine the stock; dates differ across input datasets.
The release records the prepared source file's SHA-256 in `inventory.json`.
Invalid or empty geometries are excluded; the private review folder lists their source rows and BAG IDs.
No validation beyond geometry validity, column selection, row counts and archive checksums is implied.

## Exclusions and rights

EP-Online `energy_label` and `energy_label_int` are excluded.
The [official EP-Online catalogue](https://data.overheid.nl/dataset/ep-online-webservice--geregistreerde-energielabels-)
marks the webservice as having no open licence. Public access alone does not establish permission to redistribute the bulk export.
Rows are not selected by whether a label exists.
Environmental impact results, characterisation factors and source activity identifiers are outside this release.

## Distribution and maintenance

This is the open subset prepared for dataset version 0.1 on [Zenodo concept record 8344580](https://doi.org/10.5281/zenodo.8344580).
Preparation does not imply that this version is published. Cite the version DOI after publication.
Simon van Lierde maintains the transformations in [the model repository](https://github.com/simonvanlierde/msc-thesis-ie).

From the repository root, build a new local release folder:

```sh
uv run --group dataset python scripts/build_dataset_release.py \
  --source results/geodata/BAG_buildings_with_residence_data_full.gpkg \
  --output results/dataset-v0.1
```

Validate the data package explicitly with Frictionless before depositing:

```sh
uv run --group dataset python - <<'PYTHON'
from frictionless import Package
report = Package("results/dataset-v0.1/datapackage.json").validate()
print(report.to_summary())
raise SystemExit(not report.valid)
PYTHON
```

Run `sha256sum -c SHA256SUMS` inside the output folder to verify its files.
The builder creates the sibling ZIP and never contacts Zenodo.
