from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import research_overfit as ro
import research_overfit_sensitivity as ros

ROOT = Path("data/research_backtest")
LEDGER = Path("research_trial_ledger.csv")
SIMPLE = "hyp_static_neutral_no_growth_low_vol_n10_equal_reconstructed"


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    simple = pd.read_csv(ROOT / "static_equal_monthly.csv", parse_dates=["signal_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])
    specs = pd.read_csv(ROOT / "strategy_specs.csv")

    names = specs["strategy"].dropna().astype(str).tolist()
    m = monthly[monthly["strategy"].isin(names)][["signal_date", "strategy", "net_return"]]
    wide = m.pivot(index="signal_date", columns="strategy", values="net_return").sort_index()

    sx = simple[["signal_date", "net_return"]].drop_duplicates("signal_date").set_index("signal_date")["net_return"]
    if len(sx) != 102:
        raise RuntimeError(f"Expected 102 simple-primary months, got {len(sx)}")
    wide[SIMPLE] = sx.reindex(wide.index)

    spy = bench.set_index("signal_date")["spy_return"].reindex(wide.index)
    active = wide.sub(spy, axis=0)
    active = active.dropna(axis=0, how="any")
    if len(active) != 102:
        raise RuntimeError(f"Expected 102 common months, got {len(active)}")

    sharpes = active.apply(ro.sharpe_periodic)
    ledger_floor = len(sharpes)
    if LEDGER.exists():
        ledger = pd.read_csv(LEDGER)
        mask = ledger["counts_for_trial_floor"].astype(str).str.lower().isin(["true", "1", "yes"])
        ledger_floor = max(ledger_floor, int(mask.sum()))

    simple_active = active[SIMPLE]
    simple_sr = float(sharpes[SIMPLE])
    n_grid = sorted(set([len(sharpes), ledger_floor, 25, 50, 100]))
    rows = []
    for n_trials in n_grid:
        sr0 = ros.threshold_for_n(sharpes, n_trials)
        probability = ro.dsr_probability(simple_active, sr0)
        rows.append({
            "strategy": SIMPLE,
            "matrix_candidates_including_simple": len(sharpes),
            "trial_count_assumed": n_trials,
            "ledger_trial_floor": ledger_floor,
            "active_sharpe_annualized": simple_sr * math.sqrt(12.0),
            "selection_threshold_annualized": sr0 * math.sqrt(12.0),
            "dsr_probability": probability,
            "passes_95pct": bool(np.isfinite(probability) and probability >= 0.95),
        })
    dsr = pd.DataFrame(rows)
    dsr.to_csv(ROOT / "simple_primary_dsr_sensitivity.csv", index=False)

    pbo_rows = []
    for blocks in (6, 8, 10):
        _, summary = ro.cscv_pbo(active, blocks=blocks)
        row = summary.iloc[0].to_dict()
        row["blocks"] = blocks
        pbo_rows.append(row)
    pbo = pd.DataFrame(pbo_rows)
    pbo.to_csv(ROOT / "simple_primary_pbo_sensitivity.csv", index=False)

    ledger_row = dsr[dsr["trial_count_assumed"] == ledger_floor].iloc[0]
    robust_pbo = bool((pbo["pbo"] < 0.20).all()) if len(pbo) else False
    status = pd.DataFrame([{
        "strategy": SIMPLE,
        "trial_ledger_floor": ledger_floor,
        "dsr_probability_at_ledger_floor": float(ledger_row["dsr_probability"]),
        "dsr_passes_95pct_at_ledger_floor": bool(ledger_row["passes_95pct"]),
        "pbo_below_0_20_all_block_scenarios": robust_pbo,
        "research_statistical_gate": "PASS" if bool(ledger_row["passes_95pct"]) and robust_pbo else "FAIL",
        "note": "Diagnostic only; correlated strategy trials make DSR approximate. Forward OOS remains the decisive validation path.",
    }])
    status.to_csv(ROOT / "simple_primary_overfit_status.csv", index=False)

    print(dsr.to_string(index=False), flush=True)
    print("\nPBO sensitivity:\n" + pbo.to_string(index=False), flush=True)
    print("\nStatus:\n" + status.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
