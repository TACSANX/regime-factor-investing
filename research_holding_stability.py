from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("data/research_backtest")
PRIMARY = "hyp_no_growth_low_vol_macro_sector_n10_equal"
SCORE_CONTROL = "hyp_no_growth_low_vol_macro_sector_n10_score"


def sets_by_date(df: pd.DataFrame, strategy: str) -> dict[pd.Timestamp, set[str]]:
    x = df[df["strategy"] == strategy]
    return {
        pd.Timestamp(date): set(g["Symbol"].astype(str))
        for date, g in x.groupby("signal_date")
    }


def main() -> None:
    h = pd.read_csv(
        ROOT / "holdings.csv",
        usecols=["strategy", "signal_date", "Symbol", "Security", "GICS Sector", "weight"],
        parse_dates=["signal_date"],
    )
    primary = h[h["strategy"] == PRIMARY].copy()
    if primary.empty:
        raise RuntimeError(f"No holdings for {PRIMARY}")

    psets = sets_by_date(h, PRIMARY)
    ssets = sets_by_date(h, SCORE_CONTROL)
    dates = sorted(psets)

    transitions = []
    for prev_date, date in zip(dates[:-1], dates[1:]):
        prev, cur = psets[prev_date], psets[date]
        overlap = prev & cur
        entrants = cur - prev
        exits = prev - cur
        transitions.append({
            "previous_signal_date": prev_date.date().isoformat(),
            "signal_date": date.date().isoformat(),
            "previous_count": len(prev),
            "current_count": len(cur),
            "retained_count": len(overlap),
            "retention_rate_of_current": len(overlap) / len(cur) if cur else 0.0,
            "entrants": len(entrants),
            "exits": len(exits),
        })
    t = pd.DataFrame(transitions)

    common_dates = sorted(set(psets) & set(ssets))
    mismatch_months = sum(psets[d] != ssets[d] for d in common_dates)

    summary = pd.DataFrame([{
        "strategy": PRIMARY,
        "months": len(dates),
        "transitions": len(t),
        "avg_retained_names": float(t["retained_count"].mean()),
        "median_retained_names": float(t["retained_count"].median()),
        "min_retained_names": int(t["retained_count"].min()),
        "avg_retention_rate": float(t["retention_rate_of_current"].mean()),
        "avg_new_names_per_month": float(t["entrants"].mean()),
        "max_new_names_in_month": int(t["entrants"].max()),
        "score_equal_holding_mismatch_months": int(mismatch_months),
    }])

    months = primary["signal_date"].nunique()
    freq = (
        primary.groupby(["Symbol", "Security", "GICS Sector"])
        .agg(months_held=("signal_date", "nunique"), avg_weight=("weight", "mean"))
        .reset_index()
    )
    freq["fraction_of_months_held"] = freq["months_held"] / months
    freq = freq.sort_values(["months_held", "Symbol"], ascending=[False, True])

    sector = (
        primary.groupby("GICS Sector")
        .agg(holding_observations=("Symbol", "size"), distinct_symbols=("Symbol", "nunique"))
        .reset_index()
    )
    sector["share_of_holding_observations"] = sector["holding_observations"] / len(primary)
    sector = sector.sort_values("holding_observations", ascending=False)

    summary.to_csv(ROOT / "simple_holding_churn.csv", index=False)
    t.to_csv(ROOT / "simple_holding_transitions.csv", index=False)
    freq.to_csv(ROOT / "simple_holding_frequency.csv", index=False)
    sector.to_csv(ROOT / "simple_holding_sector_frequency.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print("\nMost frequent holdings:\n" + freq.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
