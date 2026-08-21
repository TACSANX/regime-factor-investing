from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("data/research_backtest")

SHORTLIST = [
    (
        "primary",
        "hyp_no_growth_low_vol_macro_sector_n10_score",
        "Main simplified candidate: keep Value, Momentum, Quality and Sector Rotation; remove standalone Growth, Low-Vol ranking and direct Macro-Sector tilt.",
    ),
    (
        "macro_control",
        "hyp_static_neutral_no_growth_low_vol_n10",
        "Control for dynamic macro regime weighting: same simplified factor removals with fixed Neutral regime weights.",
    ),
    (
        "legacy_control",
        "full_dynamic_n10_score",
        "Original complex all-factor control retained as a reference baseline.",
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
