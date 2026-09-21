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
- `hydrotrends.describe_extended`/`ExtendedDescriptiveStats`: a ~38-field
  Descriptive Statistics Summary beyond `describe`'s 15 -- Data Quality
  (N/Missing/NF no-flow count/NNeg negative count/Record Completeness),
  Central Tendency (Mean/Median/Mode/Geometric Mean/Harmonic Mean),
  Dispersion (Std Dev/Variance/CV/IQR/Range/MAD), Estimation Uncertainty
  (Standard Error/95% normal-theory CI), Extremes (Min/Max/Last-5-Years
  Mean), Quantiles (P10/25/75/90), Distribution Shape (Skewness/Kurtosis/
  Pearson's 2nd/Bowley's/L-Skewness/Shapiro-Wilk/Anderson-Darling), Flow
  Duration (Q90/95/99 exceedance flows), and Totals. An addition beyond the
  source script; kept in its own module so `describe`'s v26-parity-tested
  numbers are untouched.
- `generate_report` writes Horizontal (one row per year, across that year's
  periods) and Vertical (one row per period, across years) Descriptive
  Statistics Summary sheets. Daily/10Daily are tripled across Cal/Hydro/Met
  Year, matching the raw Data sheets (`{Daily,10Daily[_Mean]}_Summ_{H,V}_
  {CY,HY,MY}_{unit}`, framing key abbreviated to fit Excel's 31-char
  sheet-name limit); Monthly/Hydro_Season/Met_Season stay single (Hydro- or
  Met-Year framed, matching their Data sheets, which were never tripled
  either) as `{scale}_Summary_{H,V}_{unit}`, plus a single across-years
  `Annual_Summary_{unit}` (annual volume has no within-year sub-period to
  summarise Horizontally).
- `hydrotrends.plots.generate_plots`: the plotting counterpart to
  `generate_report` -- writes a directory tree of PNG/interactive-HTML files
  instead of a workbook. Not imported by the top-level `hydrotrends` package
  (matches `hydrotrends.viz.plotting`'s existing lazy-import boundary, so
  `import hydrotrends` still doesn't pull in Matplotlib/Plotly/Seaborn).
  First piece: the source script's Section 12 (v23) "all 5 scales"
  distribution/duration/flood-exceedance suite -- a histogram+KDE grid and a
  box-whisker grid per scale (Daily split into one grid per calendar month;
  10-Daily/Monthly/Seasonal each a single grid; Annual a single histogram),
  a duration curve (static + interactive) per scale, and -- Daily/10-Daily
  only, where flood limits are physically meaningful for a flow unit -- a
  flood-exceedance overlay and heatmap (static + interactive). New plot
  functions backing it in `hydrotrends.viz.plotting`: `boxwhisker_grid`,
  `single_histogram`, `flood_overlay_static`/`flood_overlay_interactive`,
  `flood_heatmap_interactive`. `hydrotrends.core.utils.flood_limits_for_unit`
  converts the canonical 1000-Cusecs flood limits (LF/MF/HF/VHF/EHF) to an
  active flow unit.
  Second piece: the source script's Section 13 (v24) Daily-only
  per-calendar-month day-grid suite -- for every calendar month, a 3x10 grid
  (3x11 for 31-day months, extra column only in the last row) with one cell
  per calendar day showing box-whisker, histogram+KDE, violin, parametric-
  bounds (LOWESS + mean ± 3 Std Dev + Sen's slope + Bai-Perron break),
  robust-bounds (LOWESS + median ± 1.5 IQR + Sen's slope + Pettitt break),
  anomaly, ITA-scatter, decadal-blocks, or recent-vs-long-term views across
  that day's years, plus a Daily-scale duration curve (static + interactive,
  with an exceedance-probability annotation the Section-12 duration curve
  doesn't have) and flood heatmap/overlay (overlay: static only, matching the
  source). New plot functions in `hydrotrends.viz.plotting`:
  `day_grid_layout`, `build_day_grid`, `hist_mode`, the nine `draw_*_cell`
  functions, `daily_duration_curve_static`/`daily_duration_curve_interactive`,
  `daily_flood_heatmap`/`daily_flood_heatmap_interactive`,
  `daily_flood_overlay`.
  Third piece: the source script's Section 11 (v22) whole-series 6-plot
  trend suite -- Parametric Bounds, Robust Bounds, Anomalies, ITA Scatter,
  Decadal Blocks, and Sliding Windows -- run for every Annual/Seasonal/
  Monthly/10-Daily series (Volume-based, whenever the `ReportColumn` is
  volume-paired) and, by default, every one of the 366 individual calendar
  days (Flow-based) -- full source parity, with a new
  `include_daily_period_suites` parameter on `generate_plots` to opt out of
  the daily piece's very large extra cost. Also writes the aggregated
  `Mean_Shifts_Summary_<unit>.xlsx` (Decadal_Blocks/Sliding_Windows sheets)
  and the source's 5 summary overview plots: a multi-decadal
  seasonal-divergence LOWESS chart plus Seasonal/Monthly/Dekadal/Daily
  "Monotonic Trends" heatmaps condensing every period's trend direction and
  significance into one calendar-style panel each. New plot functions in
  `hydrotrends.viz.plotting`: `anomaly_bar_chart`, `compute_decadal_summary`/
  `decadal_blocks_bar_chart`, `compute_sliding_window_summary`/
  `sliding_windows_bar_chart`, `seasonal_divergence_lowess_chart`,
  `seasonal_trend_heatmap`, `monthly_trend_heatmap`, `dekadal_trend_heatmap`,
  `daily_trend_heatmap` -- distinct from the source's per-calendar-day
  Section-13 cell equivalents (`draw_anomaly_cell`/`draw_decadal_blocks_cell`/
  `draw_recent_vs_longterm_cell`), whose formulas and labels genuinely
  differ. The Dekadal/Daily summary heatmaps use the *flow* column/unit even
  though the Dekadal_Plots whole-series suite is Volume-based, and the
  Monthly summary heatmap doesn't blank a "ns" (not-significant) annotation
  the way its Seasonal/Dekadal/Daily counterparts do -- both are genuine
  inconsistencies in the source script, reproduced here for parity rather
  than "fixed".

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
- `viz.reports`'s cell-styling helper now reuses cached `Font`/`PatternFill`
  objects for a repeated (background, bold, size, color) combination instead
  of constructing a fresh, value-equal instance per cell. Cuts `generate_report`
  time roughly in half on a multi-decade daily record (~50s -> ~24s in a
  30-year synthetic benchmark) -- openpyxl deduplicates every style object it's
  handed against a workbook-wide registry, and doing that from scratch for
  thousands of value-equal-but-distinct objects was the dominant cost.

### Scope note
`hydrotrends` now has parity-tested statistical coverage (see above) for
daily/10-daily flow and monthly/seasonal/annual volume trend analysis, plus
the source script's full visualization suite (Sections 11-13 — see above)
via `hydrotrends.plots.generate_plots` — see the README's "Current scope"
section for exactly what's covered.

## [0.1.0] - 2026-07-10

### Added
- First alpha: project scaffolding, packaging, and CI.

[Unreleased]: https://github.com/shakeel2017phd/hydro-trend-analysis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shakeel2017phd/hydro-trend-analysis/releases/tag/v0.1.0# Changelog
