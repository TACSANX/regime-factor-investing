from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("data/research_backtest")

# Prefer the simpler allocation when its incremental return versus the more
# complex alternative is not statistically distinguishable. This is a research
# shortlist only; it does not promote anything to live use.
SHORTLIST = [
    (
        "primary_simple",
        "hyp_no_growth_low_vol_macro_sector_n10_equal",
        "Simplified primary research candidate: Value + Momentum + Quality + Sector Rotation, dynamic regime factor weights, no standalone Growth, no Low-Vol ranking, no direct Macro-Sector tilt, equal-weight top 10.",
    ),
    (
        "score_allocation_control",
        "hyp_no_growth_low_vol_macro_sector_n10_score",
        "Same ranking and macro definition as the primary candidate, but score-proportional sizing. Retained as the allocation control because its return advantage over equal weight is very small and statistically indistinguishable.",
    ),
    (
        "macro_control_provisional",
        "hyp_static_neutral_no_growth_low_vol_n10",
        "Provisional fixed-Neutral macro control. Allocation is still inverse-vol, so the queued same-score-allocation static trial is required for a clean macro conclusion.",
    ),
]


def main() -> None:
    summary = pd.read_csv(ROOT / "summary.csv").set_index("portfolio")
    validation = pd.read_csv(ROOT / "validation.csv").set_index("strategy")
    rows = []
    for role, strategy, interpretation in SHORTLIST:
        if strategy not in summary.index or strategy not in validation.index:
            continue
        s = summary.loc[strategy]
        v = validation.loc[strategy]
        rows.append(
            {
                "role": role,
                "strategy": strategy,
                "cagr": s.get("cagr"),
                "sharpe": s.get("sharpe_0rf"),
                "max_drawdown": s.get("max_drawdown"),
                "annualized_active_mean_vs_spy": v.get("annualized_active_mean"),
                "bootstrap_prob_active_gt_0": v.get("bootstrap_prob_active_gt_0"),
                "rolling_36_outperformance_rate": v.get("rolling_36_outperformance_rate"),
                "rolling_60_outperformance_rate": v.get("rolling_60_outperformance_rate"),
                "first_half_cagr": v.get("first_half_strategy_cagr"),
                "second_half_cagr": v.get("second_half_strategy_cagr"),
                "interpretation": interpretation,
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "shortlist.csv", index=False)


if __name__ == "__main__":
    main()
