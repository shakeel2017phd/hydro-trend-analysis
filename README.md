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

df = ht.read_series("inflow.csv", date_col="date", value_col="inflow_cumecs")
result = ht.analyze(df, value_col="inflow_cumecs")   # trend + ITA + frequency
result.summary()
```

Or from the command line:

```bash
hydro-trend analyze inflow.csv --value-col inflow_cumecs --out ./outputs
```

## Features

- Descriptive statistics across daily / 10-daily / monthly / seasonal / annual scales
- Trend tests: Mann-Kendall (incl. modified), Sen's slope, Pettitt change point
- Innovative Trend Analysis (ITA) with low/medium/high sub-trend detection
- Frequency & flow-duration (exceedance-probability) curves
- Matplotlib/Seaborn static plots and interactive Plotly HTML exports

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
