"""Frozen copy of the source script's statistical core, for parity testing only.

This is a **verbatim** copy of Section 1 ("SELF-CONTAINED STATISTICAL METHODS")
and Section 4 ("FULL STATISTICAL ANALYSIS") of
``Indus_Trend_Analysis_Daily_v26.py`` -- the reference script ``hydrotrends``
was migrated from -- plus ``compute_lowess`` from its Section 11 (visualization)
module. It exists so :mod:`test_v26_parity` can run the *original* formulas,
unmodified, against the bundled sample data and diff the result against
``hydrotrends.analyze_series`` / ``analyze_by_period`` /
``hydrotrends.viz.plotting.compute_lowess``, instead of just trusting that a
hand-audit of both implementations agrees.

Do not "clean up" this file to match the rest of the codebase's style (naming,
formatting, error handling) -- any such change would defeat its purpose. If
the source script changes, this file should be re-copied from it, not
hand-edited to drift back into sync.

The reference script's Section 2 (data loading, hardcoded to a local CSV path)
and the rest of Sections 6-12 (Excel writing, most of the Matplotlib/Plotly
plotting) are intentionally *not* included: they need a real file on disk and
a full plotting stack, and running them isn't necessary to check the
statistical formulas for parity.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SELF-CONTAINED STATISTICAL METHODS  (verbatim from the source script)
# ═══════════════════════════════════════════════════════════════════════════
def _mk_s(x: np.ndarray) -> float:
    n = len(x); s = 0.0
    for k in range(n - 1): s += np.sign(x[k + 1:] - x[k]).sum()
    return s

def _mk_var(x: np.ndarray) -> float:
    n = len(x)
    tie_counts = Counter(x.tolist())
    tp = sum(v * (v - 1) * (2 * v + 5) for v in tie_counts.values())
    return (n * (n - 1) * (2 * n + 5) - tp) / 18

def _zp(s: float, var_s: float) -> tuple:
    if var_s <= 0: return 0.0, 1.0
    z = (s - np.sign(s)) / np.sqrt(var_s)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(float(z), 6), round(float(p), 6)

def _trend_label(p: float, s: float, alpha: float = 0.05) -> str:
    if p < alpha and s > 0: return "increasing"
    if p < alpha and s < 0: return "decreasing"
    return "no trend"

def mk_original(series: pd.Series) -> dict:
    x = np.array(series.dropna(), dtype=float)
    s, var_s = _mk_s(x), _mk_var(x)
    z, p = _zp(s, var_s)
    return {"trend": _trend_label(p, s), "p": p, "z": z, "s": s}

def mk_hamed_rao(series: pd.Series, alpha: float = 0.05) -> dict:
    x = np.array(series.dropna(), dtype=float)
    n = len(x)
    if n < 10: return mk_original(pd.Series(x))
    slope, _ = sens_slope(x)
    x_detrended = x - np.arange(n) * slope
    ranks = stats.rankdata(x_detrended)
    var_s = _mk_var(x); s = _mk_s(x)
    n_ns = 1.0
    threshold = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    mean_rank = ranks.mean()
    den = np.sum((ranks - mean_rank)**2)
    for lag in range(1, n - 3):
        num = np.sum((ranks[:n-lag] - mean_rank) * (ranks[lag:] - mean_rank))
        rho = num / den if den != 0 else 0.0
        if abs(rho) >= threshold:
            n_ns += 2 * (n - lag) * (n - lag - 1) * (n - lag - 2) * rho / (n * (n - 1) * (n - 2))
    var_mod = var_s * n_ns
    z, p = _zp(s, var_mod)
    return {"trend": _trend_label(p, s, alpha), "p": p, "z": z}

def sens_slope(x: np.ndarray) -> tuple:
    n = len(x)
    t = np.arange(n, dtype=float)
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    if not slopes: return np.nan, np.nan
    sl = float(np.median(slopes))
    ic = float(np.median(x - sl * t))
    return sl, ic

def ols_fit(y: np.ndarray) -> tuple:
    n = len(y)
    t = np.arange(n, dtype=float)
    t_bar, y_bar = t.mean(), y.mean()
    sxy = ((t - t_bar) * (y - y_bar)).sum()
    sxx = ((t - t_bar) ** 2).sum()
    sst = ((y - y_bar) ** 2).sum()
    if sxx == 0: return np.nan, np.nan, np.nan, np.nan
    sl = sxy / sxx
    ic = y_bar - sl * t_bar
    sse = ((y - (sl * t + ic)) ** 2).sum()
    r2 = max(0.0, 1 - sse / sst) if sst > 0 else 0.0
    se = np.sqrt(sse / max(n - 2, 1) / sxx)
    t_stat = sl / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t_stat), df=n - 2)
    return round(sl, 8), round(float(ic), 6), round(float(r2), 6), round(float(p), 6)

def innovative_trend_analysis(series: pd.Series) -> dict:
    x = np.array(series.dropna(), dtype=float)
    n = len(x)
    if n < 4: return {"ita_slope": np.nan, "ita_trend": "no trend"}
    if n % 2 != 0: x = x[1:]; n = len(x)
    half = n // 2
    x1, x2 = x[:half], x[half:]
    ita_slope = 2 * (x2.mean() - x1.mean()) / n
    if ita_slope > 1e-6: trend = "increasing"
    elif ita_slope < -1e-6: trend = "decreasing"
    else: trend = "no trend"
    return {"ita_slope": float(ita_slope), "ita_trend": trend}

def pettitt_test(series: pd.Series) -> tuple:
    x = np.array(series.dropna(), dtype=float)
    n = len(x)
    if n < 5: return np.nan, np.nan, np.nan
    r = stats.rankdata(x)
    U = 2 * np.cumsum(r) - np.arange(1, n + 1) * (n + 1)
    abs_U = np.abs(U)
    K = abs_U.max()
    cp_idx = int(abs_U.argmax()) + 1
    if cp_idx >= n: cp_idx = n - 1
    p = min(2.0 * np.exp(-6 * K**2 / (n**3 + n**2)), 1.0)
    return cp_idx, round(K, 2), round(float(p), 4)

def cusum_cp(series: pd.Series) -> int:
    x = np.array(series.dropna(), dtype=float)
    n = len(x)
    if n < 5: return np.nan
    cusum_arr = np.cumsum(x - x.mean())
    cp_idx = int(np.argmax(np.abs(cusum_arr))) + 1
    if cp_idx >= n: cp_idx = n - 1
    return cp_idx

def bai_perron_cp(series: pd.Series, n_bkps: int = 1, min_size: int = 5) -> list:
    x = np.array(series.dropna(), dtype=float)
    n = len(x)
    if n < 2 * min_size: return []
    best_cost = np.inf; best_t = -1
    for t in range(min_size, n - min_size + 1):
        seg1, seg2 = x[:t], x[t:]
        cost = np.sum((seg1 - seg1.mean())**2) + np.sum((seg2 - seg2.mean())**2)
        if cost < best_cost:
            best_cost = cost; best_t = t
    return [best_t] if best_t != -1 else []

def sig_code(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)): return "ns"
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("* " if p < 0.05 else (". " if p < 0.10 else "ns")))

# ═══════════════════════════════════════════════════════════════════════════
# 4.  FULL STATISTICAL ANALYSIS  (verbatim from the source script)
# ═══════════════════════════════════════════════════════════════════════════
_NAN_KEYS = [
    "trend","p_value","z_score","sens_slope","sens_slope_pct","significance",
    "mmk_trend","mmk_p","mmk_z","mmk_sig","lin_slope","lin_intercept","lin_r2","lin_p",
    "log_slope","log_r2","log_p","ma5_mean","ma5_std","ma5_trend",
    "ita_slope", "ita_trend", "pettitt_cp_year","pettitt_K","pettitt_p","pettitt_sig",
    "cusum_cp_year","bpcp_year",
]

def full_analysis(series: pd.Series, years: np.ndarray = None) -> dict:
    s = series.dropna()
    n = len(s)
    yrs = np.asarray(years[:n]) if years is not None and len(years) >= n else np.arange(n)
    arr = s.values.astype(float)
    result: dict = {
        "n": n, "mean": round(float(arr.mean()), 4), "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std(ddof=1)), 4),
        "cv_pct": round(float(arr.std(ddof=1) / arr.mean() * 100), 4) if arr.mean() != 0 else np.nan,
        "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
        "p10": round(float(np.percentile(arr, 10)), 4), "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4), "p90": round(float(np.percentile(arr, 90)), 4),
        "skew": round(float(stats.skew(arr, bias=False)), 4), "kurt": round(float(stats.kurtosis(arr, bias=False)), 4),
    }

    if n < 4:
        result.update({k: np.nan for k in _NAN_KEYS})
        return result

    mv = arr.mean()
    omk = mk_original(s)
    sl, _ = sens_slope(arr)
    result.update({
        "trend": omk["trend"], "p_value": round(omk["p"], 4), "z_score": round(omk["z"], 4),
        "sens_slope": round(sl, 6) if not np.isnan(sl) else np.nan,
        "sens_slope_pct": round(sl / mv * 100, 4) if (mv and not np.isnan(sl)) else np.nan,
        "significance": sig_code(omk["p"]),
    })

    try: mmk = mk_hamed_rao(s); result.update({"mmk_trend": mmk["trend"], "mmk_p": round(mmk["p"], 4), "mmk_z": round(mmk["z"], 4), "mmk_sig": sig_code(mmk["p"])})
    except: result.update({k: np.nan for k in ("mmk_trend","mmk_p","mmk_z","mmk_sig")})

    try: sl2, ic, r2, lp = ols_fit(arr); result.update({"lin_slope": sl2, "lin_intercept": ic, "lin_r2": round(r2, 4), "lin_p": round(lp, 4)})
    except: result.update({k: np.nan for k in ("lin_slope","lin_intercept","lin_r2","lin_p")})

    try:
        pos_mask = arr > 0
        if pos_mask.sum() >= 4: lsl, _, lr2, llp = ols_fit(np.log(arr[pos_mask])); result.update({"log_slope": round(lsl, 6), "log_r2": round(lr2, 4), "log_p": round(llp, 4)})
        else: raise ValueError
    except: result.update({k: np.nan for k in ("log_slope","log_r2","log_p")})

    ma5 = s.rolling(window=5, min_periods=5).mean().dropna()
    if len(ma5) >= 3: result.update({"ma5_mean": round(float(ma5.mean()), 4), "ma5_std": round(float(ma5.std(ddof=1)), 4), "ma5_trend": mk_original(ma5)["trend"]})
    else: result.update({k: np.nan for k in ("ma5_mean","ma5_std","ma5_trend")})

    ita = innovative_trend_analysis(s)
    result.update({"ita_slope": round(ita["ita_slope"], 6) if not np.isnan(ita["ita_slope"]) else np.nan, "ita_trend": ita["ita_trend"]})

    cp_idx, K_val, p_val = pettitt_test(s)
    result.update({"pettitt_cp_year": int(yrs[int(cp_idx)]) if (not (isinstance(cp_idx, float) and np.isnan(cp_idx)) and int(cp_idx) < len(yrs)) else np.nan, "pettitt_K": K_val, "pettitt_p": p_val, "pettitt_sig": sig_code(p_val)})

    cs = cusum_cp(s)
    result["cusum_cp_year"] = int(yrs[int(cs)]) if (not (isinstance(cs, float) and np.isnan(cs)) and int(cs) < len(yrs)) else np.nan

    bp = bai_perron_cp(s, n_bkps=1)
    result["bpcp_year"] = int(yrs[bp[0]]) if (bp and bp[0] < len(yrs)) else np.nan

    return result

# ═══════════════════════════════════════════════════════════════════════════
# 11.  VISUALIZATION MODULE  (compute_lowess only, verbatim from the source script)
# ═══════════════════════════════════════════════════════════════════════════
def compute_lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> np.ndarray:
    n = len(x); y_sm = np.zeros(n); r = int(np.ceil(frac * n))
    for i in range(n):
        d = np.abs(x - x[i])
        d_max = np.sort(d)[r] if r < n else np.sort(d)[-1]
        if d_max == 0: d_max = 1.0
        w = np.clip(1 - (d / d_max)**3, 0, 1)**3
        X = np.vstack((np.ones(n), x)).T
        WX = w[:, None] * X
        XTWX = X.T @ WX
        try: beta = np.linalg.solve(XTWX, WX.T @ y); y_sm[i] = beta[0] + beta[1] * x[i]
        except np.linalg.LinAlgError: y_sm[i] = np.sum(w * y) / np.sum(w)
    return y_sm
