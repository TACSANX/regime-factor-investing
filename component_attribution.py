from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
BLOCK = 12
REPS = 20000

# Positive delta means the left-hand model outperformed the right-hand model.
COMPARISONS = [
    ("value_factor", "full_dynamic_n10_invvol", "no_value_n10", "Effect of retaining the value ranking factor"),
    ("momentum_factor", "full_dynamic_n10_invvol", "no_momentum_n10", "Effect of retaining the momentum ranking factor"),
    ("quality_factor", "full_dynamic_n10_invvol", "no_quality_n10", "Effect of retaining the original quality ranking factor"),
    ("growth_factor", "full_dynamic_n10_invvol", "no_growth_n10", "Effect of retaining the standalone growth ranking factor"),
    ("low_vol_ranking_factor", "full_dynamic_n10_invvol", "no_low_vol_n10", "Effect of retaining low-vol in ranking; both sides still use inverse-vol allocation"),
    ("sector_rotation_factor", "full_dynamic_n10_invvol", "no_sector_rotation_n10", "Effect of retaining sector-rotation ranking factor"),
    ("macro_sector_tilt", "full_dynamic_n10_invvol", "dynamic_no_macro_sector_n10", "Effect of direct macro-sector score/tilt, holding dynamic regime factor weights"),
    ("dynamic_regime_factor_weights", "dynamic_no_macro_sector_n10", "static_neutral_n10_invvol", "Effect of regime-varying factor weights after direct macro-sector tilt is removed in the base factor set"),
    ("dynamic_regime_weights_best_factor_set", "hyp_no_growth_low_vol_macro_sector_n10", "hyp_static_neutral_no_growth_low_vol_n10", "Effect of regime-varying factor weights with growth, low-vol ranking and direct macro-sector tilt all removed"),
    ("inverse_vol_allocation_base", "full_dynamic_n10_invvol", "full_dynamic_n10_equal", "Inverse-vol allocation versus equal-weight allocation under base ranking"),
    ("inverse_vol_vs_score_allocation_base", "full_dynamic_n10_invvol", "full_dynamic_n10_score", "Inverse-vol allocation versus score-proportional allocation under base ranking"),
    ("score_vs_invvol_best_factor_set", "hyp_no_growth_low_vol_macro_sector_n10_score", "hyp_no_growth_low_vol_macro_sector_n10", "Score-proportional versus inverse-vol allocation under the best-discovered factor set"),
    ("equal_vs_invvol_best_factor_set", "hyp_no_growth_low_vol_macro_sector_n10_equal", "hyp_no_growth_low_vol_macro_sector_n10", "Equal-weight versus inverse-vol allocation under the best-discovered factor set"),
    ("qmj_quality_replacement", "hyp_qmj_proxy_no_growth_low_vol_macro_sector_n10_score", "hyp_no_growth_low_vol_macro_sector_n10_score", "QMJ-inspired quality proxy versus the original quality factor, holding the discovered factor set and score allocation fixed"),
]


def circular_block_indices(n: int, block: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n / block)
    starts = rng.integers(0, n, size=(reps, blocks_needed))
    offsets = np.arange(block, dtype=int)
    return ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(reps, -1)[:, :n]


def paired_test(delta: pd.Series, label: str) -> dict:
    x = pd.Series(delta, dtype=float).dropna().to_numpy()
    if len(x) < 24:
        return {}
    seed = int(hashlib.sha1((label + str(len(x))).encode()).hexdigest()[:8], 16)
    idx = circular_block_indices(len(x), BLOCK, REPS, seed)
    annual_means = x[idx].mean(axis=1) * 12.0
    observed = float(x.mean() * 12.0)
    return {
        "months": len(x),
        "annualized_mean_return_delta": observed,
        "ci025": float(np.quantile(annual_means, 0.025)),
        "ci50": float(np.quantile(annual_means, 0.50)),
        "ci975": float(np.quantile(annual_means, 0.975)),
        "bootstrap_probability_delta_gt_0": float((annual_means > 0).mean()),
        "block_months": BLOCK,
        "bootstrap_reps": REPS,
    }


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    wide = monthly.pivot(index="signal_date", columns="strategy", values="net_return").sort_index()
    rows = []
    for component, left, right, interpretation in COMPARISONS:
        if left not in wide.columns or right not in wide.columns:
            rows.append({
                "component": component,
                "left_strategy": left,
                "right_strategy": right,
                "available": False,
                "interpretation": interpretation,
            })
            continue
        pair = wide[[left, right]].dropna()
        result = paired_test(pair[left] - pair[right], component)
        rows.append({
            "component": component,
            "left_strategy": left,
            "right_strategy": right,
            "available": True,
            **result,
            "interpretation": interpretation,
            "note": "Paired circular-block bootstrap on realized monthly net-return difference; diagnostic, not a correction for historical-universe or macro-vintage bias.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "component_attribution.csv", index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
