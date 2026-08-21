from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
DYNAMIC = "hyp_no_growth_low_vol_macro_sector_n10_equal"
STATIC_SOURCE = "hyp_static_neutral_no_growth_low_vol_n10"
STATIC_RECON = "hyp_static_neutral_no_growth_low_vol_n10_equal_reconstructed"


def cagr(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    if x.empty:
        return np.nan
    wealth = float((1.0 + x).prod())
    return wealth ** (12.0 / len(x)) - 1.0 if wealth > 0 else np.nan


def holdings_by_date(h: pd.DataFrame, strategy: str) -> dict[pd.Timestamp, pd.DataFrame]:
    x = h[h["strategy"] == strategy].copy()
    return {pd.Timestamp(d): g.copy() for d, g in x.groupby("signal_date")}


def sector_weights(g: pd.DataFrame) -> dict[str, float]:
    if g.empty:
        return {}
    n = len(g)
    return (g.groupby("GICS Sector").size() / n).to_dict()


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    static = pd.read_csv(ROOT / "static_equal_monthly.csv", parse_dates=["signal_date"])
    holdings = pd.read_csv(ROOT / "holdings.csv", parse_dates=["signal_date"])

    d = monthly[monthly["strategy"] == DYNAMIC][["signal_date", "regime", "net_return"]].rename(columns={"net_return": "dynamic_return"})
    s = static[["signal_date", "net_return"]].rename(columns={"net_return": "static_return"})
    ret = d.merge(s, on="signal_date", how="inner").sort_values("signal_date")
    if len(ret) != 102:
        raise RuntimeError(f"Expected 102 paired months, got {len(ret)}")
    ret["delta"] = ret["dynamic_return"] - ret["static_return"]
    ret["year"] = ret["signal_date"].dt.year

    regime_rows = []
    for regime, g in ret.groupby("regime"):
        regime_rows.append({
            "regime": regime,
            "months": len(g),
            "dynamic_avg_monthly_return": float(g["dynamic_return"].mean()),
            "static_avg_monthly_return": float(g["static_return"].mean()),
            "dynamic_minus_static_annualized_mean": float(g["delta"].mean() * 12.0),
            "dynamic_win_rate": float((g["delta"] > 0).mean()),
        })
    pd.DataFrame(regime_rows).to_csv(ROOT / "macro_equal_regime_delta.csv", index=False)

    year_rows = []
    for year, g in ret.groupby("year"):
        dc = cagr(g["dynamic_return"])
        sc = cagr(g["static_return"])
        year_rows.append({
            "year": int(year),
            "months": len(g),
            "dynamic_cagr": dc,
            "static_cagr": sc,
            "dynamic_minus_static_cagr": dc - sc,
        })
    pd.DataFrame(year_rows).to_csv(ROOT / "macro_equal_year_delta.csv", index=False)

    dh = holdings_by_date(holdings, DYNAMIC)
    sh = holdings_by_date(holdings, STATIC_SOURCE)
    regime_map = dict(zip(ret["signal_date"], ret["regime"]))
    overlap_rows = []
    for date in sorted(set(dh) & set(sh)):
        dg, sg = dh[date], sh[date]
        ds, ss = set(dg["Symbol"].astype(str)), set(sg["Symbol"].astype(str))
        union = ds | ss
        overlap = ds & ss
        dw, sw = sector_weights(dg), sector_weights(sg)
        sectors = set(dw) | set(sw)
        sector_half_l1 = 0.5 * sum(abs(dw.get(k, 0.0) - sw.get(k, 0.0)) for k in sectors)
        overlap_rows.append({
            "signal_date": date.date().isoformat(),
            "regime": regime_map.get(date),
            "dynamic_names": len(ds),
            "static_names": len(ss),
            "overlap_names": len(overlap),
            "jaccard": len(overlap) / len(union) if union else np.nan,
            "names_different_per_side": len(ds - ss),
            "sector_weight_half_l1": sector_half_l1,
        })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(ROOT / "macro_equal_selection_overlap.csv", index=False)

    summary = pd.DataFrame([{
        "dynamic_strategy": DYNAMIC,
        "static_strategy": STATIC_RECON,
        "months": len(overlap_df),
        "avg_overlap_names": float(overlap_df["overlap_names"].mean()),
        "median_overlap_names": float(overlap_df["overlap_names"].median()),
        "min_overlap_names": int(overlap_df["overlap_names"].min()),
        "identical_selection_months": int((overlap_df["overlap_names"] == 10).sum()),
        "avg_names_different_per_side": float(overlap_df["names_different_per_side"].mean()),
        "avg_jaccard": float(overlap_df["jaccard"].mean()),
        "avg_sector_weight_half_l1": float(overlap_df["sector_weight_half_l1"].mean()),
        "annualized_mean_return_delta": float(ret["delta"].mean() * 12.0),
    }])
    summary.to_csv(ROOT / "macro_equal_selection_summary.csv", index=False)

    by_regime = overlap_df.groupby("regime").agg(
        months=("signal_date", "size"),
        avg_overlap_names=("overlap_names", "mean"),
        avg_names_different_per_side=("names_different_per_side", "mean"),
        avg_sector_weight_half_l1=("sector_weight_half_l1", "mean"),
    ).reset_index()
    by_regime.to_csv(ROOT / "macro_equal_selection_by_regime.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print("\nReturn delta by regime:\n" + pd.DataFrame(regime_rows).to_string(index=False), flush=True)
    print("\nSelection delta by regime:\n" + by_regime.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
