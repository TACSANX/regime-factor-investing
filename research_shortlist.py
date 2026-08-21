from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("data/research_backtest")

# Research shortlist only. Prefer the simpler model when a more complex model's
# incremental return is not statistically distinguishable.
CORE = [
    (
        "macro_timing_control",
        "hyp_no_growth_low_vol_macro_sector_n10_equal",
        "Same simplified factor set and equal-weight top 10, but with dynamic regime factor weights. Kept as the macro-timing control; its paired return advantage over the static model is not statistically distinguishable.",
    ),
    (
        "score_allocation_control",
        "hyp_no_growth_low_vol_macro_sector_n10_score",
        "Dynamic-regime version with score-proportional sizing. Kept only as an allocation control; score versus equal weighting is statistically indistinguishable.",
    ),
]


def row_from_main(summary: pd.DataFrame, validation: pd.DataFrame, role: str, strategy: str, interpretation: str) -> dict | None:
    if strategy not in summary.index or strategy not in validation.index:
        return None
    s, v = summary.loc[strategy], validation.loc[strategy]
    return {
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


def main() -> None:
    summary = pd.read_csv(ROOT / "summary.csv").set_index("portfolio")
    validation = pd.read_csv(ROOT / "validation.csv").set_index("strategy")
    rows: list[dict] = []

    static_summary_path = ROOT / "static_equal_summary.csv"
    static_validation_path = ROOT / "static_equal_validation.csv"
    if static_summary_path.exists() and static_validation_path.exists():
        ss = pd.read_csv(static_summary_path).iloc[0]
        sv = pd.read_csv(static_validation_path).iloc[0]
        rows.append({
            "role": "primary_simple",
            "strategy": ss["strategy"],
            "cagr": ss["cagr"],
            "sharpe": ss["sharpe_0rf"],
            "max_drawdown": ss["max_drawdown"],
            "annualized_active_mean_vs_spy": sv["annualized_active_mean"],
            "bootstrap_prob_active_gt_0": sv["bootstrap_prob_active_gt_0"],
            "rolling_36_outperformance_rate": sv["rolling_36_outperformance_rate"],
            "rolling_60_outperformance_rate": sv["rolling_60_outperformance_rate"],
            "first_half_cagr": sv["first_half_strategy_cagr"],
            "second_half_cagr": sv["second_half_strategy_cagr"],
            "interpretation": "Simplest current research candidate: Value + Momentum + Quality + Sector Rotation with fixed Neutral factor weights, no standalone Growth, no Low-Vol ranking, no direct Macro-Sector tilt, and equal-weight top 10. Dynamic macro weighting has not shown a statistically distinguishable incremental return.",
        })

    for role, strategy, interpretation in CORE:
        row = row_from_main(summary, validation, role, strategy, interpretation)
        if row is not None:
            rows.append(row)

    pd.DataFrame(rows).to_csv(ROOT / "shortlist.csv", index=False)


if __name__ == "__main__":
    main()
