from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("data/research_backtest")
SOURCE_STRATEGY = "hyp_static_neutral_no_growth_low_vol_n10"
DYNAMIC_EQUAL = "hyp_no_growth_low_vol_macro_sector_n10_equal"
OUTPUT_STRATEGY = "hyp_static_neutral_no_growth_low_vol_n10_equal_reconstructed"
COST_BPS = 10.0
MIN_VALID_WEIGHT = 0.95


def yahoo_symbol(symbol: str) -> str:
    return str(symbol).replace(".", "-")


def _extract_open(data: pd.DataFrame, symbol: str) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        if "Open" not in data.columns.get_level_values(0):
            return pd.Series(dtype=float)
        x = data["Open"]
        if isinstance(x, pd.DataFrame):
            if symbol in x.columns:
                s = x[symbol]
            elif x.shape[1] == 1:
                s = x.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
        else:
            s = x
    else:
        if "Open" not in data:
            return pd.Series(dtype=float)
        s = data["Open"]
    s = pd.Series(s, dtype=float).dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def download_opens(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, batch_size: int = 40) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        data = yf.download(
            batch,
            start=(start - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            if "Open" in data.columns.get_level_values(0):
                parts.append(data["Open"])
        elif len(batch) == 1 and "Open" in data:
            parts.append(pd.DataFrame({batch[0]: data["Open"]}, index=data.index))
    if not parts:
        raise RuntimeError("Yahoo returned no open-price data")
    x = pd.concat(parts, axis=1).sort_index()
    x = x.loc[:, ~x.columns.duplicated()]
    x.index = pd.to_datetime(x.index).tz_localize(None)
    return x


def required_price_dates(holdings: pd.DataFrame, periods: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    period_map = periods.set_index("signal_date")[["execution_date", "exit_execution_date"]].to_dict("index")
    required: dict[str, set[pd.Timestamp]] = {}
    for row in holdings[["signal_date", "Symbol"]].itertuples(index=False):
        p = period_map.get(row.signal_date)
        if not p:
            continue
        ys = yahoo_symbol(row.Symbol)
        required.setdefault(ys, set()).update({pd.Timestamp(p["execution_date"]), pd.Timestamp(p["exit_execution_date"])})
    return required


def _has_required(open_px: pd.DataFrame, symbol: str, dates: set[pd.Timestamp]) -> bool:
    if symbol not in open_px.columns:
        return False
    for date in dates:
        if date not in open_px.index:
            return False
        value = open_px.at[date, symbol]
        if not np.isfinite(value):
            return False
    return True


def repair_missing_opens(
    open_px: pd.DataFrame,
    required: dict[str, set[pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    x = open_px.copy()
    missing = [s for s, dates in required.items() if not _has_required(x, s, dates)]
    print(f"Yahoo symbols requiring individual repair: {len(missing)}", flush=True)
    for pos, symbol in enumerate(missing, start=1):
        for attempt in range(1, retries + 1):
            try:
                data = yf.download(
                    symbol,
                    start=(start - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                    end=(end + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=False,
                )
                s = _extract_open(data, symbol)
                if not s.empty:
                    x = x.reindex(x.index.union(s.index)).sort_index()
                    if symbol not in x.columns:
                        x[symbol] = np.nan
                    x.loc[s.index, symbol] = s.values
                if _has_required(x, symbol, required[symbol]):
                    break
            except Exception as exc:
                print(f"Retry {attempt}/{retries} failed for {symbol}: {exc}", flush=True)
            time.sleep(float(attempt))
        if pos % 25 == 0:
            print(f"Yahoo individual repair: {pos}/{len(missing)}", flush=True)
    unresolved = [s for s, dates in required.items() if not _has_required(x, s, dates)]
    return x, unresolved


def price_return(open_px: pd.DataFrame, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    ys = yahoo_symbol(symbol)
    if ys not in open_px.columns:
        return np.nan
    try:
        p0 = float(open_px.loc[start, ys])
        p1 = float(open_px.loc[end, ys])
    except Exception:
        return np.nan
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
        return np.nan
    return p1 / p0 - 1.0


def turnover(target: dict[str, float], prev_end: dict[str, float]) -> tuple[float, float]:
    if not prev_end:
        return 1.0, 1.0
    half_l1 = 0.5 * sum(abs(target.get(s, 0.0) - prev_end.get(s, 0.0)) for s in set(target) | set(prev_end))
    return float(half_l1), float(2.0 * half_l1)


def end_weights(target: dict[str, float], returns: dict[str, float]) -> dict[str, float]:
    raw = {s: target[s] * (1.0 + returns.get(s, 0.0)) for s in target}
    den = float(sum(raw.values()))
    return {s: v / den for s, v in raw.items()} if den > 0 else dict(target)


def perf_stats(r: pd.Series) -> dict:
    x = pd.Series(r, dtype=float).dropna()
    eq = (1.0 + x).cumprod()
    years = len(x) / 12.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(x.std(ddof=1) * math.sqrt(12.0))
    sharpe = float(x.mean() / x.std(ddof=1) * math.sqrt(12.0)) if x.std(ddof=1) > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    return {
        "months": len(x),
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe_0rf": sharpe,
        "max_drawdown": float(dd.min()),
        "positive_month_rate": float((x > 0).mean()),
    }


def circular_block_means(values: np.ndarray, block: int = 12, reps: int = 20000, seed: int = 20260821) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=float)
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    for i in range(reps):
        starts = rng.integers(0, n, size=blocks_needed)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        out[i] = float(values[idx].mean())
    return out


def main() -> None:
    holdings = pd.read_csv(ROOT / "holdings.csv", parse_dates=["signal_date", "execution_date"])
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date", "execution_date", "exit_execution_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])

    h = holdings[holdings["strategy"] == SOURCE_STRATEGY].copy()
    periods = monthly[monthly["strategy"] == SOURCE_STRATEGY][
        ["signal_date", "execution_date", "exit_execution_date", "regime"]
    ].drop_duplicates("signal_date").sort_values("signal_date")
    if h.empty or periods.empty:
        raise RuntimeError("Static-neutral source holdings/monthly rows are missing")

    symbols = sorted({yahoo_symbol(s) for s in h["Symbol"].astype(str)})
    start = periods["execution_date"].min()
    end = periods["exit_execution_date"].max()
    open_px = download_opens(symbols, start, end)
    required = required_price_dates(h, periods)
    open_px, unresolved = repair_missing_opens(open_px, required, start, end)
    if unresolved:
        raise RuntimeError(f"Yahoo still missing required price dates for {len(unresolved)} symbols: {unresolved[:20]}")

    prev_end: dict[str, float] = {}
    rows: list[dict] = []
    cost_rate = COST_BPS / 10000.0

    for p in periods.itertuples(index=False):
        picks = h[h["signal_date"] == p.signal_date].copy()
        names = picks["Symbol"].astype(str).tolist()
        if not names:
            continue
        target = {s: 1.0 / len(names) for s in names}
        half_l1, traded = turnover(target, prev_end)
        indiv = {s: price_return(open_px, s, p.execution_date, p.exit_execution_date) for s in names}
        indiv = {s: r for s, r in indiv.items() if np.isfinite(r)}
        valid_weight = float(sum(target[s] for s in indiv))
        if valid_weight < MIN_VALID_WEIGHT:
            raise RuntimeError(f"Unexpected incomplete month {p.signal_date.date()}: valid weight {valid_weight:.1%}")
        gross = float(sum(target[s] * indiv[s] for s in indiv) / valid_weight)
        cost = float(traded * cost_rate)
        net = gross - cost
        prev_end = end_weights(target, indiv)
        rows.append({
            "strategy": OUTPUT_STRATEGY,
            "signal_date": p.signal_date,
            "execution_date": p.execution_date,
            "exit_execution_date": p.exit_execution_date,
            "regime": p.regime,
            "gross_return": gross,
            "net_return": net,
            "turnover_half_l1": half_l1,
            "traded_notional": traded,
            "cost": cost,
            "valid_weight_fraction": valid_weight,
            "names": len(names),
        })

    out = pd.DataFrame(rows)
    if len(out) != len(periods):
        raise RuntimeError(f"Reconstructed {len(out)} of {len(periods)} periods; refusing to publish incomplete results")
    out.to_csv(ROOT / "static_equal_monthly.csv", index=False)

    b = bench[["signal_date", "spy_return"]].copy()
    merged = out.merge(b, on="signal_date", how="inner")
    if len(merged) != len(out):
        raise RuntimeError(f"Benchmark merge produced {len(merged)} of {len(out)} periods")
    active = merged["net_return"] - merged["spy_return"]
    summary = perf_stats(out["net_return"])
    summary.update({
        "strategy": OUTPUT_STRATEGY,
        "annualized_active_mean_vs_spy": float(active.mean() * 12.0),
        "avg_monthly_traded_notional": float(out["traded_notional"].mean()),
        "avg_valid_weight_fraction": float(out["valid_weight_fraction"].mean()),
    })
    pd.DataFrame([summary]).to_csv(ROOT / "static_equal_summary.csv", index=False)

    dynamic = monthly[monthly["strategy"] == DYNAMIC_EQUAL][["signal_date", "net_return"]].rename(columns={"net_return": "dynamic_return"})
    pair = dynamic.merge(out[["signal_date", "net_return"]].rename(columns={"net_return": "static_return"}), on="signal_date", how="inner")
    if len(pair) != len(out):
        raise RuntimeError(f"Dynamic/static merge produced {len(pair)} of {len(out)} periods")
    d = (pair["dynamic_return"] - pair["static_return"]).to_numpy(dtype=float)
    boot = circular_block_means(d)
    pairwise = pd.DataFrame([{
        "comparison": "dynamic_equal_vs_static_neutral_equal",
        "left_strategy": DYNAMIC_EQUAL,
        "right_strategy": OUTPUT_STRATEGY,
        "months": len(d),
        "annualized_mean_return_delta": float(np.mean(d) * 12.0),
        "ci025": float(np.quantile(boot, 0.025) * 12.0),
        "ci50": float(np.quantile(boot, 0.50) * 12.0),
        "ci975": float(np.quantile(boot, 0.975) * 12.0),
        "bootstrap_probability_dynamic_gt_static": float(np.mean(boot > 0.0)),
        "block_months": 12,
        "bootstrap_reps": len(boot),
    }])
    pairwise.to_csv(ROOT / "macro_equal_pairwise.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print(pairwise.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
