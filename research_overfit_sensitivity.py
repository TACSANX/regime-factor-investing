from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import research_overfit as ro

ROOT = Path("data/research_backtest")
LEDGER = Path("research_trial_ledger.csv")


def threshold_for_n(sharpes: pd.Series, n_trials: int) -> float:
    s = pd.Series(sharpes, dtype=float).dropna()
    if n_trials <= 1 or len(s) <= 1:
        return 0.0
    sigma = float(s.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    gamma = 0.5772156649015329
    nd = NormalDist()
    return sigma * (
        (1.0 - gamma) * nd.inv_cdf(1.0 - 1.0 / n_trials)
        + gamma * nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])
    specs = pd.read_csv(ROOT / "strategy_specs.csv")
    names = specs["strategy"].dropna().astype(str).tolist()
    m = monthly[monthly["strategy"].isin(names)][["signal_date", "strategy", "net_return"]]
    wide = m.pivot(index="signal_date", columns="strategy", values="net_return").sort_index()
    spy = bench.set_index("signal_date")["spy_return"].reindex(wide.index)
    active = wide.sub(spy, axis=0)
    sharpes = active.apply(ro.sharpe_periodic)

    ledger_floor = len(names)
    if LEDGER.exists():
        ledger = pd.read_csv(LEDGER)
        mask = ledger["counts_for_trial_floor"].astype(str).str.lower().isin(["true", "1", "yes"])
        ledger_floor = max(ledger_floor, int(mask.sum()))

    observed_best = sharpes.idxmax()
    best_active = active[observed_best].dropna()
    n_grid = sorted(set([len(names), ledger_floor, 25, 50, 100]))
    dsr_rows = []
    for n_trials in n_grid:
        sr0 = threshold_for_n(sharpes, n_trials)
        dsr_rows.append({
            "strategy": observed_best,
            "current_matrix_candidates": len(names),
            "trial_count_assumed": n_trials,
            "ledger_trial_floor": ledger_floor,
            "observed_active_sharpe_annualized": float(sharpes[observed_best] * math.sqrt(12)),
            "selection_threshold_annualized": float(sr0 * math.sqrt(12)),
            "dsr_probability": ro.dsr_probability(best_active, sr0),
            "passes_95pct": bool(ro.dsr_probability(best_active, sr0) >= 0.95),
        })
    pd.DataFrame(dsr_rows).to_csv(ROOT / "dsr_trial_count_sensitivity.csv", index=False)

    pbo_rows = []
    for blocks in (6, 8, 10):
        detail, summary = ro.cscv_pbo(active, blocks=blocks)
        row = summary.iloc[0].to_dict()
        row["blocks"] = blocks
        pbo_rows.append(row)
    pbo_df = pd.DataFrame(pbo_rows)
    pbo_df.to_csv(ROOT / "pbo_block_sensitivity.csv", index=False)

    robust_pbo = bool((pbo_df["pbo"] < 0.20).all()) if len(pbo_df) else False
    dsr95 = bool(pd.DataFrame(dsr_rows)["passes_95pct"].all()) if dsr_rows else False
    pd.DataFrame([{
        "observed_best_strategy": observed_best,
        "trial_ledger_floor": ledger_floor,
        "dsr_passes_95pct_all_trial_scenarios": dsr95,
        "pbo_below_0_20_all_block_scenarios": robust_pbo,
        "robust_overfit_gate": "PASS" if dsr95 and robust_pbo else "FAIL",
        "note": "Sensitivity analysis only. A pass would still not resolve point-in-time constituent, delisted-price, or macro-vintage bias.",
    }]).to_csv(ROOT / "overfit_sensitivity_status.csv", index=False)

    print(pd.DataFrame(dsr_rows).to_string(index=False), flush=True)
    print("\n" + pbo_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
