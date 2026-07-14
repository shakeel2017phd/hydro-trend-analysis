# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure (`core`, `data`, `stats`, `viz`).
- Migration of the Indus/Tarbela daily-inflow analysis script into modules.
- Monthly and cropping-seasonal/annual volume aggregation (`monthly_volumes`,
  `hydro_seasonal_volumes`) and trend/descriptive analysis
  (`analyze_monthly_volumes`, `analyze_hydro_seasonal_volumes`,
  `describe_monthly_volumes`, `describe_hydro_seasonal_volumes`), matching the
  source script's `monthly_vol` / `season_dfs` computations.
- Meteorological-season volume aggregation and analysis
  (`met_seasonal_volumes`, `analyze_met_seasonal_volumes`,
  `describe_met_seasonal_volumes`) — Winter/Spring/Summer/Monsoon/Autumn, an
  addition beyond the source script, using the previously-unused
  `Config.season_schemes` (`METEOROLOGICAL`) option.
- `generate_report` writes `Monthly Trends`/`Hydro Season Trends`/
  `Met Season Trends` (and descriptive) sheets whenever a `ReportColumn` is
  paired with a volume column, gated by a new `season_schemes` parameter
  (both schemes on by default, matching `Config`).
- The `hydro-trend` CLI pairs every flow unit with its volume unit via
  `UNIT_PAIRS` by default, and now writes **one workbook per flow/volume unit
  pairing** (e.g. `..._Cusecs_MAF.xlsx` / `..._Cumecs_BCM.xlsx`), restoring
  the source script's two-workbook-per-run deliverable instead of bundling
  every unit into one file.
- `tests/test_v26_parity.py` runs a frozen, verbatim copy of the source
  script's statistical core (`tests/reference_v26_core.py`) against
  `hydrotrends.analyze_preprocessed` on identical synthetic data, asserting
  numeric equality on every shared descriptive/trend/change-point field for
  daily and 10-daily resolutions.
- `hydrotrends.viz.plotting.parametric_bounds_plot` / `robust_bounds_plot`:
  the source script's "Parametric Bounds" (LOWESS + mean ± 3 Std Dev + Sen's
  slope + Bai-Perron break) and "Robust Bounds" (LOWESS + median ± 1.5 IQR +
  Sen's slope + Pettitt break) plots, plus the `compute_lowess` smoothing
  function they depend on — parity-tested against the source in
  `tests/test_plotting.py`.
- Meteorological-year (`MetYear`, Dec 1 → Nov 30) framing alongside the
  existing hydrological/calendar-year framings, with matching period/month/
  dekad orderings (`MET_PERIODS`/`MET_MONTHS`/`MET_DEKADS`) and
  `PreprocessedData.met`.
- Raw-value **Data** sheets in `generate_report`: `Daily_Data_{Cal,Hydro,Met}
  _Year_{unit}`, the 10-Daily-derived equivalents, `Monthly_Data_{unit}`,
  `Hydro_Season_Data_{unit}`, `Met_Season_Data_{unit}`, and
  `Annual_Data_{unit}` — one pivot per Cal/Hydro/Met Year framing, each with
  full row-wise and column-wise descriptive statistics (mirroring the source
  script's `write_period_data_sheet`/`write_monthly_data_sheet`/
  `write_seasonal_data_sheet`/`write_annual_data_sheet`, extended to also
  cover the meteorological-year framing the source script doesn't have).

### Removed
- The `Descriptive (...)`/`Monthly Descriptive (...)`/`Hydro Season
  Descriptive (...)`/`Met Season Descriptive (...)` sheets, `generate_report`'s
  `include_descriptive` parameter, and the CLI's `--no-descriptive` flag —
  superseded by the new Data sheets above, which carry the same descriptive
  statistics plus the raw values they're computed from.

### Fixed
- `analyze_preprocessed`'s daily/10-daily per-period tables now sort by and
  report the calendar `Year` column for change-point mapping, matching the
  source script exactly (it previously used `HydroYear`, which is off by one
  from the source's reported change-point year for any period falling in
  Jan/Feb/Mar). Trend-test results were unaffected — `Year` and `HydroYear`
  differ by the same constant offset for every row of a given period, so sort
  order (and therefore every trend test) was already identical either way.

### Scope note
`hydrotrends` now has parity-tested statistical coverage (see above) for
daily/10-daily flow and monthly/seasonal/annual volume trend analysis, plus a
growing subset of the source script's visualization suite (see above). Still
not ported: per-month box-whisker/histogram/violin grids, decadal blocks, and
flood-exceedance overlays, across all five aggregation scales — see the
README's "Current scope" section for exactly what's covered.

## [0.1.0] - 2026-07-10

### Added
- First alpha: project scaffolding, packaging, and CI.

[Unreleased]: https://github.com/shakeel2017phd/hydro-trend-analysis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shakeel2017phd/hydro-trend-analysis/releases/tag/v0.1.0# Changelog
