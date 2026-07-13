# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure (`core`, `data`, `stats`, `viz`).
- Migration of the Indus/Tarbela daily-inflow analysis script into modules.
- Monthly and seasonal/annual volume aggregation (`monthly_volumes`,
  `seasonal_volumes`) and trend/descriptive analysis (`analyze_monthly_volumes`,
  `analyze_seasonal_volumes`, `describe_monthly_volumes`,
  `describe_seasonal_volumes`), matching the source script's `monthly_vol` /
  `season_dfs` computations. `generate_report` and the `hydro-trend` CLI now
  write the corresponding `Monthly Trends`/`Seasonal Trends` (and descriptive)
  sheets whenever a `ReportColumn` is paired with a volume column — the CLI
  pairs every flow unit with its volume unit via `UNIT_PAIRS` by default.

## [0.1.0] - 2026-07-10

### Added
- First alpha: project scaffolding, packaging, and CI.

[Unreleased]: https://github.com/shakeel2017phd/hydro-trend-analysis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shakeel2017phd/hydro-trend-analysis/releases/tag/v0.1.0# Changelog
