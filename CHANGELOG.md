# Changelog

Changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Contributor guidance, issue forms and security reporting instructions.
- A task runner and a cold-install check for the documented development setup.
- Dependency licence checks and OpenSSF Scorecard reporting.
- A dashboard accessibility statement and Content Security Policy.
- A 2,000-building regression fixture with real geometry and independent synthetic energy labels.
- An unpublished open-input dataset builder with schemas, attribution and checksums.
- A Python 3.15 runtime, a two-city (Oegstgeest, Voorschoten) pipeline prototype and a sensitivity and packaging profiling script.
- Dashboard data tables for neighbourhoods and hourly heatwave values.

### Changed

- The urban heat island correction cut status quo cooling demand from 1,118 to 527 GWh; the committed `data/output/` files still hold the 2023 results.

### Fixed

- Keep the pinned paper-numbers test offline even when upstream raw caches are stale.
- Keep temporal data available when `ResizeObserver` is unavailable.
- Reject profiler output paths inside the dataset packaging directory before writing files.
- Notebooks now track their input CSVs and write under the configured results root.
- Dashboard maps retry after a data failure and ignore obsolete style responses; narrow screens no longer overflow.
- Pipeline runs now become stale after script, helper or locked dependency changes.
- Thermodynamic edits no longer rebuild unrelated building geodata.

[Unreleased]: https://github.com/simonvanlierde/msc-thesis-ie/commits/HEAD
