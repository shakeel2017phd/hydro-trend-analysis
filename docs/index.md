# hydro-trend-analysis

Professional hydrological time-series analysis: descriptive statistics, trend
detection, Innovative Trend Analysis (ITA), change-point detection, frequency /
flow-duration analysis, and publication-ready visualization and Excel reporting.

> **Import name** `hydrotrends` · **Distribution (PyPI) name** `hydro-trend-analysis`

## Install

```bash
git clone https://github.com/shakeel2017phd/hydro-trend-analysis.git
cd hydro-trend-analysis
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quickstart

```python
import hydrotrends as ht

# Bundled sample (or read your own — see "Reading your data" below)
pre = ht.datasets.load_preprocessed("tarbela_daily")

# One-call analysis + formatted Excel report
ht.generate_report(
    pre, "report.xlsx",
    columns=[ht.ReportColumn(ht.COL_FLOW_CUSECS, "Cusecs")],
    title="Tarbela Daily Inflow — Trend Analysis",
)
```

Or from the command line:

```bash
hydro-trend analyze inflow.csv -o report.xlsx -r daily --value-column inflow_cusec
```

A runnable tour lives in [`examples/quickstart.py`](../examples/quickstart.py).

## Reading your data

Every input is described by an `InputSpec`, so daily and 10-daily files — even
with different headers, units, or date encodings — can be processed together:

```python
daily = ht.InputSpec("tarbela_daily.csv", "daily")            # DD/MM/YYYY, cusecs

tendaily = ht.InputSpec(
    "tarbela_10daily.csv", "10daily",
    value_column="Inflow_1000Cusecs",   # different header
    value_scale=1000,                    # 1000-Cusecs -> Cusecs (avoids a 1000x error)
    date_format="dekad_compact",         # "2020Apr1" -> a real date
)

df = ht.read_input(daily)                # -> canonical Date + Inflow_Cusecs frame
df, report = ht.clean(df)                # optional: policy-driven cleaning
pre = ht.preprocess(df, "daily")         # derived columns, volumes, seasons, filtering
```

`date_format` options: `auto` (day-first), `month_first` (US), `iso`, and
`dekad_compact` (10-daily). `value_scale` multiplies the value column on read.

## Key concepts

- **Temporal resolution.** `daily` (per-day) or `10daily` (dekad averages). A
  10-daily value is an *average* inflow over its dekad, so volumes are weighted
  by the dekad's day-count — a daily and a 10-daily file of the same flow give
  the same annual volume.
- **Hydrological (water) year.** Runs 1 April – 31 March; January–March belong to
  the previous water year. Incomplete years at the start/end of the record are
  dropped automatically.
- **Seasons.** Both classifications are produced: **cropping** (Early/Late Kharif
  with the day-10 June split, and Rabi) and **meteorological** (`MET_SEASONS`).
- **Per-period trends.** For each period (day-of-year or dekad) the analysis runs
  across all years, so a slope is per year.

## What it computes

| Area | Functions |
|---|---|
| Descriptive | `describe`, `describe_by` |
| Trend tests | `mann_kendall`, `mann_kendall_modified` (Hamed-Rao), `sens_slope`, `linear_regression`, `log_linear_regression` |
| Innovative Trend Analysis | `innovative_trend_analysis` (+ low/medium/high sub-trends) |
| Change-point | `pettitt_test`, `cusum_change_point`, `bai_perron_change_point` |
| Frequency | `flow_duration_curve`, `exceedance_counts`, `flow_percentiles` (Q10/Q50/Q90) |
| Orchestration | `analyze_series`, `analyze_by_period`, `analyze_preprocessed`, `generate_report` |

Every trend test follows a soft-guard convention: on a series too short to be
meaningful it returns `NaN` / "no trend" rather than raising (trend tests need
≥ 4 years of record per period).

## Plotting

Plotting lives in `hydrotrends.viz.plotting` and is imported separately, so a
plain `import hydrotrends` stays lightweight (no Matplotlib/Plotly pulled in):

```python
from hydrotrends.viz import plotting

fig = plotting.timeseries_interactive(pre.hydro, value_col=ht.COL_FLOW_CUSECS)
fig.write_html("inflow.html", include_plotlyjs="cdn")
```

Available: `timeseries_static` / `timeseries_interactive`, `ita_scatter`,
`flow_duration_curve_static` / `_interactive`, `trend_scatter`,
`distribution_grid`, `flood_heatmap`.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest            # 86 tests
ruff check . && ruff format --check .
mypy src
```

## License

MIT — see [LICENSE](../LICENSE).