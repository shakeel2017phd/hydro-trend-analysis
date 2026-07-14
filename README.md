# hydro-trend-analysis

Professional hydrological time-series analysis: descriptive statistics, trend
analysis, Innovative Trend Analysis (ITA), frequency analysis, and publication-
ready visualization and reporting.

> Import name is `hydrotrends`; the distribution (PyPI) name is
> `hydro-trend-analysis`.

## Installation

```bash
# from source (until first PyPI release)
git clone https://github.com/shakeel2017phd/hydro-trend-analysis.git
cd hydro-trend-analysis
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

```python
import hydrotrends as ht

pre = ht.preprocess(
    ht.read_input(ht.InputSpec("inflow.csv", "daily", value_column="inflow_cusec")),
    ht.TimeResolution.DAILY,
)
report = ht.generate_report(
    pre,
    "report.xlsx",
    columns=[
        ht.ReportColumn(
            ht.COL_FLOW_CUSECS, "Cusecs",
            volume_column=ht.COL_VOL_MAF, volume_unit_label="MAF",
        ),
    ],
)
```

See `examples/quickstart.py` for a runnable end-to-end walkthrough (uses a
bundled sample, so it needs no external file).

Or from the command line:

```bash
hydro-trend analyze inflow.csv --value-column inflow_cusec -o ./outputs
```

## Current scope

`hydrotrends` is an active migration of a monolithic reference script
(`Indus_Trend_Analysis_Daily_v26.py`) into a tested, typed package. As of now:

- **Implemented and parity-tested** against the reference script's formulas
  (see `tests/test_v26_parity.py`): descriptive statistics; Mann-Kendall
  (original + Hamed-Rao modified); Sen's slope; linear/log-linear regression;
  5-yr backward moving average; Innovative Trend Analysis; Pettitt/CUSUM/
  Bai-Perron change-point detection — across daily and 10-daily periods, and
  monthly / cropping-seasonal (Kharif/Rabi) / meteorological-seasonal /
  annual total-volume aggregates. A CLI run writes one Excel workbook per
  flow/volume unit pairing (Cusecs+MAF, Cumecs+BCM), matching the reference
  script's two-workbook deliverable.
- **Beyond the reference script**: a ~38-field Descriptive Statistics Summary
  (`hydrotrends.describe_extended`) — Data Quality, Central Tendency
  (incl. Geometric/Harmonic Mean), Dispersion, Estimation Uncertainty (95% CI),
  Extremes, Quantiles, Distribution Shape (incl. Shapiro-Wilk/Anderson-Darling
  normality diagnostics), Flow Duration, and Totals — written as Horizontal
  (across periods within a year) and Vertical (across years for a period)
  summary sheets for every scale.
- **Not yet ported**: the reference script's full visualization suite (per
  calendar-month box-whisker/histogram/violin grids, decadal blocks,
  flood-exceedance overlays, across all five aggregation scales).
  `hydrotrends.viz.plotting` currently covers time-series plots, ITA scatter,
  flow-duration curves (with flood-limit annotation), a generic trend-scatter
  overlay, the parametric/robust LOWESS bounds plots (mean ± 3 Std Dev /
  median ± 1.5 IQR, with Sen's-slope and change-point overlays — parity-tested
  in `tests/test_plotting.py`), a histogram+KDE distribution grid, and a
  flood-exceedance heatmap — a meaningful subset, not full parity.

If you're comparing output against the reference script and the numbers
don't match, check whether the mismatch is in a feature listed above as "not
yet ported" before assuming a bug in the ported statistics.

## Features

- Descriptive statistics and trend tests across daily / 10-daily / monthly /
  seasonal / annual scales
- Trend tests: Mann-Kendall (incl. modified), Sen's slope, Pettitt/CUSUM/
  Bai-Perron change-point detection
- Innovative Trend Analysis (ITA) with low/medium/high sub-trend detection
- Frequency & flow-duration (exceedance-probability) curves
- Formatted Excel reports, one workbook per flow/volume unit pairing
- Matplotlib/Seaborn static plots and interactive Plotly HTML exports for the
  subset of chart types ported so far (see "Current scope" above)

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

Versioning is tag-driven via `setuptools_scm` — the version comes from the
latest `git` tag, not a hand-edited file.

## License

MIT — see [LICENSE](LICENSE).# hydro-trend-analysis
