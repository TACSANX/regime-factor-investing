from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
AUDIT = Path("data/research")
OUT = ROOT
BASE = "full_dynamic_n10_invvol"
BENCHMARK = "SPY"
BLOCK = 12
REPS = 10000


def perf_stats(r: pd.Series) -> dict[str, float]:
    r = pd.Series(r, dtype=float).dropna()
    if r.empty:
        return {}
    eq = (1.0 + r).cumprod()
    years = len(r) / 12.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    std = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    dd = eq / eq.cummax() - 1.0
    return {
        "cagr": cagr,
        "annual_volatility": std * math.sqrt(12) if np.isfinite(std) else np.nan,
        "sharpe_0rf": float(r.mean() / std * math.sqrt(12)) if np.isfinite(std) and std > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "months": int(len(r)),
    }


def circular_block_indices(n: int, block: int, reps: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 0), dtype=int)
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n / block)
    starts = rng.integers(0, n, size=(reps, blocks_needed))
    offsets = np.arange(block, dtype=int)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(reps, -1)[:, :n]


def bootstrap_active(active: pd.Series, block: int = BLOCK, reps: int = REPS) -> dict[str, float]:
    x = pd.Series(active, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < max(24, block * 2):
        return {}
    seed = int(hashlib.sha1(pd.util.hash_pandas_object(pd.Series(x)).values.tobytes()).hexdigest()[:8], 16)
    idx = circular_block_indices(n, block, reps, seed)
    means = x[idx].mean(axis=1) * 12.0
    observed = float(x.mean() * 12.0)
    return {
        "annualized_active_mean": observed,
        "block_months": block,
        "bootstrap_reps": reps,
        "active_mean_ci025": float(np.quantile(means, 0.025)),
        "active_mean_ci50": float(np.quantile(means, 0.50)),
        "active_mean_ci975": float(np.quantile(means, 0.975)),
        "bootstrap_prob_active_gt_0": float((means > 0).mean()),
    }


def reality_check(active_wide: pd.DataFrame, block: int = BLOCK, reps: int = REPS) -> dict[str, float | str]:
    x = active_wide.dropna(axis=0, how="any").to_numpy(dtype=float)
    if x.shape[0] < max(24, block * 2) or x.shape[1] < 2:
        return {}
    observed_means = x.mean(axis=0)
    best_i = int(np.argmax(observed_means))
    observed_max = float(observed_means[best_i])

    # Null of no positive expected active return.  Center each strategy while
    # preserving cross-strategy correlation, then resample identical circular
    # time blocks across all variants.  This is a White-style max-mean proxy,
    # not a full implementation of White's Reality Check / SPA.
    centered = x - observed_means[None, :]
    idx = circular_block_indices(len(x), block, reps, seed=20260821)
    boot_max = np.empty(reps, dtype=float)
    for i in range(reps):
        boot_max[i] = centered[idx[i], :].mean(axis=0).max()
    p = float((boot_max >= observed_max).mean())
    return {
        "candidate_count": int(x.shape[1]),
        "common_months": int(x.shape[0]),
        "observed_best_strategy": str(active_wide.columns[best_i]),
        "observed_best_annualized_active_mean": observed_max * 12.0,
        "block_months": block,
        "bootstrap_reps": reps,
        "max_mean_reality_check_proxy_p": p,
        "passes_5pct": bool(p < 0.05),
        "note": "Circular-block max-mean data-snooping proxy; conservative diagnostic, not formal White/SPA inference.",
    }


def rolling_summary(strategy: pd.DataFrame, benchmark: pd.DataFrame, window: int) -> tuple[pd.DataFrame, dict]:
    x = strategy[["signal_date", "net_return"]].merge(
        benchmark[["signal_date", "spy_return"]], on="signal_date", how="inner"
    ).sort_values("signal_date")
    rows = []
    for end_i in range(window - 1, len(x)):
        g = x.iloc[end_i - window + 1:end_i + 1]
        years = window / 12.0
        sg = float((1.0 + g["net_return"]).prod())
        bg = float((1.0 + g["spy_return"]).prod())
        s_cagr = sg ** (1.0 / years) - 1.0
        b_cagr = bg ** (1.0 / years) - 1.0
        rows.append({
            "window_months": window,
            "start": g["signal_date"].iloc[0],
            "end": g["signal_date"].iloc[-1],
            "strategy_cagr": s_cagr,
            "spy_cagr": b_cagr,
            "excess_cagr": s_cagr - b_cagr,
        })
    df = pd.DataFrame(rows)
    summary = {
        "window_months": window,
        "windows": int(len(df)),
        "pct_windows_outperformed_spy": float((df["excess_cagr"] > 0).mean()) if len(df) else np.nan,
        "min_excess_cagr": float(df["excess_cagr"].min()) if len(df) else np.nan,
        "median_excess_cagr": float(df["excess_cagr"].median()) if len(df) else np.nan,
        "max_excess_cagr": float(df["excess_cagr"].max()) if len(df) else np.nan,
    }
    return df, summary


def half_sample(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, float]:
    x = strategy[["signal_date", "net_return"]].merge(
        benchmark[["signal_date", "spy_return"]], on="signal_date", how="inner"
    ).sort_values("signal_date")
    if len(x) < 24:
        return {}
    split = len(x) // 2
    out = {}
    for label, g in [("first_half", x.iloc[:split]), ("second_half", x.iloc[split:])]:
        s = perf_stats(g["net_return"])
        b = perf_stats(g["spy_return"])
        out[f"{label}_strategy_cagr"] = s.get("cagr", np.nan)
        out[f"{label}_spy_cagr"] = b.get("cagr", np.nan)
        out[f"{label}_excess_cagr"] = s.get("cagr", np.nan) - b.get("cagr", np.nan)
        out[f"{label}_active_arith_annual"] = float((g["net_return"] - g["spy_return"]).mean() * 12.0)
    return out


def read_bool(path: Path, column: str) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        if df.empty or column not in df.columns:
            return False
        v = df[column].iloc[0]
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes"}
        return bool(v)
    except Exception:
        return False


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    benchmark = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])
    specs = pd.read_csv(ROOT / "strategy_specs.csv")

    research_names = specs["strategy"].astype(str).tolist()
    validation_rows = []
    bootstrap_rows = []
    rolling_rows = []
    rolling_summary_rows = []
    active_columns = {}

    for name in research_names:
        g = monthly[monthly["strategy"] == name].sort_values("signal_date")
        x = g[["signal_date", "net_return"]].merge(
            benchmark[["signal_date", "spy_return"]], on="signal_date", how="inner"
        )
        if len(x) < 24:
            continue
        active = x["net_return"] - x["spy_return"]
        bs = bootstrap_active(active)
        bootstrap_rows.append({"strategy": name, **bs})
        active_columns[name] = pd.Series(active.to_numpy(), index=x["signal_date"])

        rs = {}
        for window in (36, 60):
            detail, smry = rolling_summary(g, benchmark, window)
            if not detail.empty:
                detail.insert(0, "strategy", name)
                rolling_rows.append(detail)
            smry["strategy"] = name
            rolling_summary_rows.append(smry)
            rs[f"rolling_{window}_outperformance_rate"] = smry["pct_windows_outperformed_spy"]

        halves = half_sample(g, benchmark)
        # Deliberately demanding promotion gate. It is diagnostic only; a pass
        # does not override unresolved point-in-time universe / macro-vintage bias.
        statistical_pass = bool(
            bs
            and bs.get("active_mean_ci025", -np.inf) > 0
            and rs.get("rolling_36_outperformance_rate", 0) >= 0.60
            and rs.get("rolling_60_outperformance_rate", 0) >= 0.60
            and halves.get("first_half_excess_cagr", -np.inf) > 0
            and halves.get("second_half_excess_cagr", -np.inf) > 0
        )
        validation_rows.append({
            "strategy": name,
            **bs,
            **rs,
            **halves,
            "statistical_stability_gate": statistical_pass,
        })

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    validation = pd.DataFrame(validation_rows)
    rolling_detail = pd.concat(rolling_rows, ignore_index=True) if rolling_rows else pd.DataFrame()
    rolling_smry = pd.DataFrame(rolling_summary_rows)

    active_wide = pd.concat(active_columns, axis=1).sort_index() if active_columns else pd.DataFrame()
    rc = reality_check(active_wide)
    pd.DataFrame([rc]).to_csv(OUT / "reality_check_proxy.csv", index=False)

    universe_ok = read_bool(AUDIT / "historical_universe_quality.csv", "usable_for_survivorship_free_test")
    price_ok = read_bool(AUDIT / "historical_price_coverage_summary.csv", "usable_without_alternate_price_source")
    reality_ok = bool(rc.get("passes_5pct", False))

    validation["historical_universe_gate"] = universe_ok
    validation["historical_price_gate"] = price_ok
    validation["multiple_testing_gate"] = reality_ok
    validation["production_validated"] = (
        validation["statistical_stability_gate"]
        & validation["historical_universe_gate"]
        & validation["historical_price_gate"]
        & validation["multiple_testing_gate"]
    )
    validation["status"] = np.where(validation["production_validated"], "PASS", "RESEARCH_ONLY")

    bootstrap_df.to_csv(OUT / "bootstrap_significance.csv", index=False)
    validation.to_csv(OUT / "validation.csv", index=False)
    rolling_detail.to_csv(OUT / "rolling_windows.csv", index=False)
    rolling_smry.to_csv(OUT / "rolling_summary.csv", index=False)

    pd.DataFrame([{
        "base_strategy": BASE,
        "historical_universe_usable": universe_ok,
        "historical_price_coverage_usable": price_ok,
        "multiple_testing_proxy_pass_5pct": reality_ok,
        "any_strategy_production_validated": bool(validation["production_validated"].any()),
        "decision": "RESEARCH_ONLY" if not bool(validation["production_validated"].any()) else "CANDIDATE_PASSED_ALL_GATES",
        "note": "Passing statistical diagnostics is insufficient while point-in-time universe or data-vintage gates fail.",
    }]).to_csv(OUT / "validation_status.csv", index=False)

    cols = [
        "strategy", "annualized_active_mean", "active_mean_ci025", "active_mean_ci975",
        "bootstrap_prob_active_gt_0", "rolling_36_outperformance_rate",
        "rolling_60_outperformance_rate", "first_half_excess_cagr", "second_half_excess_cagr",
        "statistical_stability_gate", "production_validated", "status",
    ]
    print("\n=== RESEARCH VALIDATION ===\n" + validation[cols].to_string(index=False), flush=True)
    print("\n=== MULTIPLE-TESTING PROXY ===\n" + pd.DataFrame([rc]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
