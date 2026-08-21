from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("data/research_backtest")


def cagr(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    if x.empty:
        return np.nan
    w = float((1 + x).prod())
    return w ** (12.0 / len(x)) - 1.0 if w > 0 else np.nan


def main() -> None:
    detail = pd.read_csv(ROOT / "static_equal_contribution_detail.csv", parse_dates=["signal_date"])
    stress = pd.read_csv(ROOT / "static_equal_winner_stress_monthly.csv", parse_dates=["signal_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])[["signal_date", "spy_return"]]

    detail["year"] = detail["signal_date"].dt.year
    detail["is_it"] = detail["GICS Sector"].eq("Information Technology")
    detail["weighted_contribution"] = detail["stock_return"] / 10.0

    yearly_rows = []
    for year, g in detail.groupby("year"):
        it = g[g["is_it"]]
        yearly_rows.append({
            "year": int(year),
            "holding_observations": len(g),
            "it_holding_observations": len(it),
            "it_holding_share": float(len(it) / len(g)) if len(g) else np.nan,
            "gross_contribution_all": float(g["weighted_contribution"].sum()),
            "gross_contribution_it": float(it["weighted_contribution"].sum()),
            "it_fraction_of_positive_gross_contribution": float(
                it["weighted_contribution"].clip(lower=0).sum()
                / g["weighted_contribution"].clip(lower=0).sum()
            ) if g["weighted_contribution"].clip(lower=0).sum() > 0 else np.nan,
        })
    yearly = pd.DataFrame(yearly_rows)

    base = stress[stress["stress_case"].eq("baseline_rebuilt")][["signal_date", "net_return"]].rename(columns={"net_return": "base_return"})
    no_it = stress[stress["stress_case"].eq("exclude_information_technology")][["signal_date", "net_return"]].rename(columns={"net_return": "no_it_return"})
    paths = base.merge(no_it, on="signal_date").merge(bench, on="signal_date")
    paths["year"] = paths["signal_date"].dt.year

    perf_rows = []
    for year, g in paths.groupby("year"):
        perf_rows.append({
            "year": int(year),
            "months": len(g),
            "base_compounded_return": float((1 + g["base_return"]).prod() - 1),
            "no_it_compounded_return": float((1 + g["no_it_return"]).prod() - 1),
            "spy_compounded_return": float((1 + g["spy_return"]).prod() - 1),
            "no_it_minus_spy": float((1 + g["no_it_return"]).prod() - (1 + g["spy_return"]).prod()),
            "base_minus_no_it": float((1 + g["base_return"]).prod() - (1 + g["no_it_return"]).prod()),
        })
    perf = pd.DataFrame(perf_rows)
    out = yearly.merge(perf, on="year", how="left")
    out.to_csv(ROOT / "static_equal_it_dependence_yearly.csv", index=False)

    summary = pd.DataFrame([{
        "months": len(paths),
        "it_holding_share_all": float(detail["is_it"].mean()),
        "base_cagr": cagr(paths["base_return"]),
        "no_it_cagr": cagr(paths["no_it_return"]),
        "spy_cagr": cagr(paths["spy_return"]),
        "years_no_it_beats_spy": int((out["no_it_minus_spy"] > 0).sum()),
        "years_total": int(out["no_it_minus_spy"].notna().sum()),
        "years_it_adds_return": int((out["base_minus_no_it"] > 0).sum()),
        "peak_it_holding_share": float(out["it_holding_share"].max()),
        "peak_it_holding_share_year": int(out.loc[out["it_holding_share"].idxmax(), "year"]),
    }])
    summary.to_csv(ROOT / "static_equal_it_dependence_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
