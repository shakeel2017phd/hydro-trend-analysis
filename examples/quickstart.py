"""hydrotrends quickstart — a runnable end-to-end tour.

Uses the bundled Tarbela sample, so it runs with no external files::

    python examples/quickstart.py

It writes an Excel report and a couple of plots into ``./quickstart_output/``.
"""

from __future__ import annotations

from pathlib import Path

import hydrotrends as ht

OUT = Path("quickstart_output")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # 1. Load a bundled sample, already read + preprocessed.
    #    (For your own data: ht.read_input(ht.InputSpec("inflow.csv", "daily"))
    #     then ht.clean(...) then ht.preprocess(...).)
    pre = ht.datasets.load_preprocessed("tarbela_daily")
    print(
        f"Loaded daily sample: {len(pre.hydro):,} rows across "
        f"{pre.hydro['HydroYear'].nunique()} complete hydro years."
    )

    # 2. Descriptive statistics for the whole record.
    stats = ht.describe(pre.hydro[ht.COL_FLOW_CUSECS])
    print(
        f"Mean inflow: {stats.mean:,.0f} Cusecs | "
        f"CV: {stats.cv_pct:.1f}% | skew: {stats.skewness:.2f}"
    )

    # 3. Run every trend test on one series (the annual mean here).
    annual = pre.hydro.groupby("HydroYear")[ht.COL_FLOW_CUSECS].mean()
    mk = ht.mann_kendall(annual)
    sen = ht.sens_slope(annual)
    ita = ht.innovative_trend_analysis(annual)
    print(
        f"Annual-mean trend: {mk.trend} (p={mk.p_value:.3g}), "
        f"Sen's slope {sen.slope:.1f}/yr, ITA slope {ita.slope:.3f}"
    )

    # 4. Flow-duration indices.
    q = ht.flow_percentiles(pre.hydro[ht.COL_FLOW_CUSECS])
    print(
        f"Q90 (low flow) {q['Q90']:,.0f} | Q50 {q['Q50']:,.0f} | "
        f"Q10 (high flow) {q['Q10']:,.0f} Cusecs"
    )

    # 5. Full per-period analysis + a formatted Excel report.
    report = ht.generate_report(
        pre,
        OUT / "tarbela_report.xlsx",
        columns=[
            ht.ReportColumn(ht.COL_FLOW_CUSECS, "Cusecs"),
            ht.ReportColumn(ht.COL_FLOW_CUMECS, "Cumecs"),
        ],
        title="Tarbela Daily Inflow — Trend Analysis",
        subtitle="Mann-Kendall | Sen's Slope | ITA | Change-Point",
    )
    print(f"Report written: {report}")

    # 6. Plots (imported here so a plain `import hydrotrends` stays light).
    from hydrotrends.viz import plotting

    ts = plotting.timeseries_static(
        pre.hydro,
        value_col=ht.COL_FLOW_CUSECS,
        title="Tarbela Daily Inflow",
        y_label="Inflow (Cusecs)",
    )
    ts.savefig(OUT / "timeseries.png", dpi=150, bbox_inches="tight")

    fdc = plotting.flow_duration_curve_static(
        pre.hydro[ht.COL_FLOW_CUSECS], unit_label="Cusecs", title="Flow-Duration Curve"
    )
    fdc.savefig(OUT / "flow_duration.png", dpi=150, bbox_inches="tight")

    interactive = plotting.timeseries_interactive(
        pre.hydro, value_col=ht.COL_FLOW_CUSECS, title="Tarbela Daily Inflow"
    )
    interactive.write_html(str(OUT / "timeseries.html"), include_plotlyjs="cdn")

    print(
        f"Plots written to {OUT}/ (timeseries.png, flow_duration.png, timeseries.html)"
    )


if __name__ == "__main__":
    main()
