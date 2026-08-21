from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as bt
import screener as sc
from portfolio_constraints import build_portfolio_strict

# Observation-date lags chosen to be conservative relative to typical release
# timing. Daily market/rate data are delayed one day; monthly reference-period
# series are delayed into the following month.
MACRO_LAG_DAYS_V2 = {
    "sahm": 38,
    "curve_10y3m": 1,
    "hy_oas": 1,
    "nfci": 7,
    "real10y": 1,
    "fedfunds": 1,
    "unrate": 38,
    "cpi": 45,
    "cfnai": 60,
}


def _all_frames(duration: dict, instant: dict, debt_frames: dict) -> list[pd.DataFrame]:
    return [x for x in list(duration.values()) + list(instant.values()) + list(debt_frames.values()) if not x.empty]


def _snapshot_for_cutoff(symbol: str, cutoff: pd.Timestamp, duration: dict, instant: dict, debt_frames: dict) -> dict:
    rec = {"Symbol": symbol}
    for key, df in duration.items():
        cur, prev = sc.ttm_from_records(bt.asof_records(df, cutoff))
        rec[f"{key}_ttm"], rec[f"{key}_prior_ttm"] = cur, prev
    for key, df in instant.items():
        rec[key] = sc.latest_instant(bt.asof_records(df, cutoff))
    rec["debt"] = bt.debt_asof(debt_frames, cutoff)
    return rec


def point_in_time_fundamentals_fast(universe: pd.DataFrame, dates: list[pd.Timestamp], cache: sc.Cache) -> pd.DataFrame:
    symbols_key = "|".join(sorted(universe["Symbol"].astype(str)))
    digest = hashlib.sha1(symbols_key.encode()).hexdigest()[:12]
    panel_path = cache.root / f"fund_panel_v2_{dates[0].date()}_{dates[-1].date()}_{digest}.pkl.gz"
    if panel_path.exists():
        try:
            panel = pd.read_pickle(panel_path, compression="gzip")
            if not panel.empty:
                print(f"Loaded cached point-in-time panel: {panel_path}", flush=True)
                return panel
        except Exception:
            pass

    cik_map = sc.ticker_cik_map(cache, 87600)
    rows: list[dict] = []
    total = len(universe)
    date_index = pd.DatetimeIndex(dates)

    for pos, u in enumerate(universe.itertuples(index=False), start=1):
        symbol = str(u.Symbol)
        cik = cik_map.get(symbol.upper())
        if cik is None:
            continue
        facts = sc.fetch_company_facts(cik, cache, 87600)
        if not facts:
            continue

        duration = {k: sc.records_for_alternatives(facts, tags) for k, tags in sc.DURATION_TAGS.items()}
        instant = {k: sc.records_for_alternatives(facts, tags) for k, tags in sc.INSTANT_TAGS.items()}
        debt_frames = {tag: sc._records_for_tag(facts, "us-gaap", tag) for tag in [
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "LongTermDebtCurrent",
            "LongTermDebtNoncurrent",
            "ShortTermBorrowings",
            "DebtCurrent",
        ]}

        filings = []
        for frame in _all_frames(duration, instant, debt_frames):
            if "filed" in frame:
                filings.extend(pd.to_datetime(frame["filed"], errors="coerce").dropna().tolist())
        filings = pd.DatetimeIndex(sorted(set(filings)))

        # Fundamentals only change when a new filing becomes available.  Map
        # 103 monthly dates to the much smaller set of relevant filing cutoffs.
        cutoff_for_date: dict[pd.Timestamp, pd.Timestamp | None] = {}
        unique_cutoffs: set[pd.Timestamp] = set()
        for date in date_index:
            available = filings[filings <= date]
            cutoff = pd.Timestamp(available[-1]) if len(available) else None
            cutoff_for_date[pd.Timestamp(date)] = cutoff
            if cutoff is not None:
                unique_cutoffs.add(cutoff)

        snapshots = {
            cutoff: _snapshot_for_cutoff(symbol, cutoff, duration, instant, debt_frames)
            for cutoff in sorted(unique_cutoffs)
        }
        for date in date_index:
            cutoff = cutoff_for_date[pd.Timestamp(date)]
            rec = {"date": pd.Timestamp(date), "Symbol": symbol}
            if cutoff is not None:
                rec.update({k: v for k, v in snapshots[cutoff].items() if k != "Symbol"})
            rows.append(rec)

        if pos % 50 == 0:
            print(f"SEC point-in-time fundamentals v2: {pos}/{total}", flush=True)

    panel = pd.DataFrame(rows)
    panel.to_pickle(panel_path, compression="gzip")
    print(f"Cached point-in-time panel: {panel_path}", flush=True)
    return panel


def download_ohlcv(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opens, closes, volumes = [], [], []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        data = yf.download(
            batch,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            opens.append(data["Open"] if "Open" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index))
            closes.append(data["Close"] if "Close" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index))
            volumes.append(data["Volume"] if "Volume" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index))
        else:
            opens.append(pd.DataFrame({batch[0]: data["Open"]}, index=data.index))
            closes.append(pd.DataFrame({batch[0]: data["Close"]}, index=data.index))
            volumes.append(pd.DataFrame({batch[0]: data.get("Volume")}, index=data.index))
    if not closes:
        raise RuntimeError("Price download returned no data")

    def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
        x = pd.concat(parts, axis=1).sort_index().loc[:, lambda y: ~y.columns.duplicated()]
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return x

    return combine(opens), combine(closes), combine(volumes)


def next_session(index: pd.DatetimeIndex, after: pd.Timestamp) -> pd.Timestamp | None:
    candidates = index[index > after]
    return pd.Timestamp(candidates[0]) if len(candidates) else None


def price_return_at_open(open_px: pd.DataFrame, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if symbol not in open_px.columns:
        return np.nan
    s = open_px[symbol]
    try:
        p0, p1 = float(s.loc[start]), float(s.loc[end])
    except Exception:
        return np.nan
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
        return np.nan
    return p1 / p0 - 1.0


def perf_stats(r: pd.Series) -> dict:
    r = pd.Series(r, dtype=float).dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 12
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * math.sqrt(12))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    return {
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe_0rf": sharpe,
        "sortino_0rf": float(r.mean() / downside * math.sqrt(12)) if np.isfinite(downside) and downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "positive_month_rate": float((r > 0).mean()),
        "months": len(r),
    }


def run(args) -> None:
    config = sc.load_config(args.config)
    cache = sc.Cache(Path(config.get("cache_dir", ".cache_sp500_screener")))
    universe = sc.get_universe(cache, 87600, None)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc).date())

    symbols = universe["Yahoo Symbol"].tolist() + ["SPY", "RSP"] + list(config["sector_etfs"].values())
    open_px, close, volume = download_ohlcv(
        list(dict.fromkeys(symbols)), start - pd.Timedelta(days=430), end, int(config.get("price_batch_size", 80))
    )
    dates = bt.month_ends(close.index, start, end)
    if len(dates) < 13:
        raise RuntimeError("Need at least 13 monthly observations")
    signal_dates = dates[:-1]
    print(f"Backtest v2: {signal_dates[0].date()} to {dates[-1].date()}, {len(signal_dates)} periods", flush=True)

    fred = {name: sc.fred_series(cache, name, sid, 87600) for name, sid in sc.FRED_SERIES.items()}
    bt.MACRO_LAG_DAYS = MACRO_LAG_DAYS_V2.copy()
    fund_panel = point_in_time_fundamentals_fast(universe, signal_dates, cache)
    fund_panel["date"] = pd.to_datetime(fund_panel["date"])
    yahoo = dict(zip(universe["Symbol"], universe["Yahoo Symbol"]))

    cost_rate = args.cost_bps / 10000.0
    prev_end: dict[str, float] = {}
    perf: list[dict] = []
    holdings: list[dict] = []

    for i, signal_date in enumerate(signal_dates):
        next_signal = dates[i + 1]
        exec_start = next_session(open_px.index, signal_date)
        exec_end = next_session(open_px.index, next_signal)
        if exec_start is None or exec_end is None:
            continue

        tech = sc.price_features(close.loc[:signal_date], volume.loc[:signal_date], universe)
        sector = sc.sector_rotation(close.loc[:signal_date], universe, tech, config["sector_etfs"])
        fund = fund_panel[fund_panel["date"] == signal_date].drop(columns="date")
        macro = bt.macro_asof(fred, signal_date)
        ranked = sc.score_stocks(sc.build_features(universe, tech, fund, sector), macro, config)
        port = build_portfolio_strict(ranked, args.portfolio_n, args.max_stock_weight, args.max_sector_weight)
        if port.empty:
            continue

        target = dict(zip(port["Symbol"], port["weight"]))
        turnover = 1.0 if not prev_end else 0.5 * sum(
            abs(target.get(s, 0.0) - prev_end.get(s, 0.0)) for s in set(target) | set(prev_end)
        )
        traded_notional = 1.0 if not prev_end else 2.0 * turnover

        indiv = {}
        for sym in target:
            ys = yahoo.get(sym)
            if ys:
                ret = price_return_at_open(open_px, ys, exec_start, exec_end)
                if np.isfinite(ret):
                    indiv[sym] = ret
        valid_w = sum(target[s] for s in indiv)
        if valid_w <= 0:
            continue
        gross = sum(target[s] * indiv[s] for s in indiv) / valid_w
        cost = traded_notional * cost_rate
        net = gross - cost

        spy_ret = price_return_at_open(open_px, "SPY", exec_start, exec_end)
        rsp_ret = price_return_at_open(open_px, "RSP", exec_start, exec_end)

        raw_end = {s: target[s] * (1 + indiv.get(s, 0.0)) for s in target}
        den = sum(raw_end.values())
        prev_end = {s: v / den for s, v in raw_end.items()} if den > 0 else target

        perf.append({
            "signal_date": signal_date,
            "execution_date": exec_start,
            "next_signal_date": next_signal,
            "exit_execution_date": exec_end,
            "strategy_return": net,
            "gross_return": gross,
            "spy_return": spy_ret,
            "rsp_return": rsp_ret,
            "turnover_half_l1": turnover,
            "traded_notional": traded_notional,
            "cost": cost,
            "regime": macro.regime,
            "holdings": len(port),
            "max_stock_weight_realized": float(port["weight"].max()),
            "max_sector_weight_realized": float(port.groupby("GICS Sector")["weight"].sum().max()),
        })
        for _, h in port.iterrows():
            holdings.append({
                "signal_date": signal_date,
                "execution_date": exec_start,
                "Symbol": h["Symbol"],
                "Security": h["Security"],
                "GICS Sector": h["GICS Sector"],
                "score": h["score"],
                "weight": h["weight"],
            })
        print(
            f"{signal_date.date()} {macro.regime:16s} net={net:+.2%} SPY={spy_ret:+.2%} "
            f"turnover={turnover:.1%} sector_max={port.groupby('GICS Sector')['weight'].sum().max():.1%}",
            flush=True,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    perf_df = pd.DataFrame(perf)
    if perf_df.empty:
        raise RuntimeError("No valid backtest periods")
    perf_df["strategy_equity"] = (1 + perf_df["strategy_return"]).cumprod()
    perf_df["spy_equity"] = (1 + perf_df["spy_return"]).cumprod()
    perf_df["rsp_equity"] = (1 + perf_df["rsp_return"]).cumprod()

    s_stats = perf_stats(perf_df["strategy_return"])
    spy_stats = perf_stats(perf_df["spy_return"])
    rsp_stats = perf_stats(perf_df["rsp_return"])
    active = perf_df["strategy_return"] - perf_df["spy_return"]
    s_stats["avg_monthly_turnover_half_l1"] = float(perf_df["turnover_half_l1"].mean())
    s_stats["avg_monthly_traded_notional"] = float(perf_df["traded_notional"].mean())
    s_stats["annualized_excess_return_arith_vs_spy"] = float(active.mean() * 12)
    s_stats["information_ratio_vs_spy"] = float(active.mean() / active.std(ddof=1) * math.sqrt(12)) if active.std(ddof=1) > 0 else np.nan
    summary = pd.DataFrame([
        {"portfolio": "regime_factor_v2", **s_stats},
        {"portfolio": "SPY", **spy_stats},
        {"portfolio": "RSP", **rsp_stats},
    ])

    summary.to_csv(out / "summary.csv", index=False)
    perf_df.to_csv(out / "equity_curve.csv", index=False)
    pd.DataFrame(holdings).to_csv(out / "holdings.csv", index=False)
    pd.DataFrame([
        ["universe", "Current S&P 500 constituents; survivorship bias remains"],
        ["signal", "Month-end close/volume; signal computed after close"],
        ["execution", "Next trading session adjusted open; removes same-close execution bias"],
        ["fundamental_timing", "Only SEC facts filed on/before signal date; snapshots change only on filing dates"],
        ["macro_timing", f"Reference-date lags: {MACRO_LAG_DAYS_V2}"],
        ["transaction_cost", f"{args.cost_bps} bps per dollar traded; initial=1x NAV, later traded_notional=2*half-L1 turnover"],
        ["portfolio_constraints", f"strict max stock={args.max_stock_weight:.1%}, max sector={args.max_sector_weight:.1%}"],
        ["known_bias", "Latest-vintage FRED revision bias remains; historical constituent bias remains"],
    ], columns=["assumption", "value"]).to_csv(out / "assumptions.csv", index=False)

    print("\n=== BACKTEST V2 SUMMARY ===\n" + summary.to_string(index=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--portfolio-n", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--max-stock-weight", type=float, default=0.12)
    p.add_argument("--max-sector-weight", type=float, default=0.30)
    p.add_argument("--output-dir", default="data/backtest_v2")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
