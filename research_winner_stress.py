from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
COST_RATE = 10.0 / 10000.0


def cagr(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    if x.empty:
        return np.nan
    wealth = float((1.0 + x).prod())
    return wealth ** (12.0 / len(x)) - 1.0 if wealth > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    wealth = (1.0 + x).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min()) if not wealth.empty else np.nan


def sharpe(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd * math.sqrt(12.0)) if sd > 0 else np.nan


def end_weights(target: dict[str, float], returns: dict[str, float]) -> dict[str, float]:
    raw = {s: target[s] * (1.0 + returns[s]) for s in target}
    den = float(sum(raw.values()))
    return {s: v / den for s, v in raw.items()} if den > 0 else dict(target)


def traded_notional(target: dict[str, float], previous_end: dict[str, float]) -> float:
    if not previous_end:
        return 1.0
    return float(sum(abs(target.get(s, 0.0) - previous_end.get(s, 0.0)) for s in set(target) | set(previous_end)))


def simulate(detail: pd.DataFrame, excluded_symbols: set[str] | None = None, excluded_sector: str | None = None) -> pd.DataFrame:
    excluded_symbols = excluded_symbols or set()
    rows = []
    prev_end: dict[str, float] = {}
    for date, g in detail.groupby("signal_date", sort=True):
        x = g[~g["Symbol"].isin(excluded_symbols)].copy()
        if excluded_sector is not None:
            x = x[x["GICS Sector"] != excluded_sector]
        if x.empty:
            raise RuntimeError(f"Stress leaves no holdings at {date}")
        names = x["Symbol"].astype(str).tolist()
        target = {s: 1.0 / len(names) for s in names}
        returns = dict(zip(x["Symbol"].astype(str), x["stock_return"].astype(float)))
        gross = float(sum(target[s] * returns[s] for s in names))
        traded = traded_notional(target, prev_end)
        net = gross - traded * COST_RATE
        prev_end = end_weights(target, returns)
        rows.append({
            "signal_date": date,
            "net_return": net,
            "gross_return": gross,
            "traded_notional": traded,
            "names": len(names),
        })
    return pd.DataFrame(rows)


def main() -> None:
    detail = pd.read_csv(ROOT / "static_equal_contribution_detail.csv", parse_dates=["signal_date"])
    symbols = pd.read_csv(ROOT / "static_equal_symbol_contribution.csv")
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])[["signal_date", "spy_return"]]
    official = pd.read_csv(ROOT / "static_equal_monthly.csv", parse_dates=["signal_date"])[["signal_date", "net_return"]]

    top_symbols = symbols.head(5)["Symbol"].astype(str).tolist()
    cases = [
        ("baseline_rebuilt", set(), None),
        ("exclude_expost_top1", set(top_symbols[:1]), None),
        ("exclude_expost_top3", set(top_symbols[:3]), None),
        ("exclude_expost_top5", set(top_symbols[:5]), None),
        ("exclude_information_technology", set(), "Information Technology"),
    ]

    rows = []
    paths = []
    for label, excluded, sector in cases:
        sim = simulate(detail, excluded, sector)
        merged = sim.merge(bench, on="signal_date", how="inner")
        if len(merged) != 102:
            raise RuntimeError(f"{label}: expected 102 months, got {len(merged)}")
        rows.append({
            "stress_case": label,
            "excluded_symbols": ";".join(sorted(excluded)),
            "excluded_sector": sector or "",
            "months": len(merged),
            "avg_names": float(sim["names"].mean()),
            "min_names": int(sim["names"].min()),
            "avg_monthly_traded_notional": float(sim["traded_notional"].mean()),
            "cagr": cagr(sim["net_return"]),
            "spy_cagr": cagr(merged["spy_return"]),
            "cagr_excess_vs_spy": cagr(sim["net_return"]) - cagr(merged["spy_return"]),
            "sharpe_0rf": sharpe(sim["net_return"]),
            "max_drawdown": max_drawdown(sim["net_return"]),
        })
        paths.append(sim.assign(stress_case=label))

    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "static_equal_winner_stress.csv", index=False)
    pd.concat(paths, ignore_index=True).to_csv(ROOT / "static_equal_winner_stress_monthly.csv", index=False)

    baseline = pd.concat([official.rename(columns={"net_return": "official"}), paths[0][["signal_date", "net_return"]].rename(columns={"net_return": "rebuilt"})], axis=1)
    # Compare by position only after date equality is confirmed.
    a = official.sort_values("signal_date").reset_index(drop=True)
    b = paths[0].sort_values("signal_date").reset_index(drop=True)
    if not a["signal_date"].equals(b["signal_date"]):
        raise RuntimeError("Baseline reconstruction dates do not match official static-equal dates")
    max_abs = float((a["net_return"] - b["net_return"]).abs().max())
    if max_abs > 1e-10:
        raise RuntimeError(f"Baseline stress reconstruction mismatch: max abs {max_abs}")

    print(summary.to_string(index=False), flush=True)
    print(f"Baseline reconstruction max abs difference: {max_abs:.3e}", flush=True)


if __name__ == "__main__":
    main()
