from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import screener as sc

# Conservative publication lags. These prevent obvious same-period look-ahead.
# FRED still supplies latest-vintage history, so macro revision bias remains.
MACRO_LAG_DAYS = {
    "sahm": 7, "curve_10y3m": 1, "hy_oas": 1, "nfci": 7,
    "real10y": 1, "fedfunds": 1, "unrate": 7, "cpi": 20, "cfnai": 30,
}


def asof_records(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "filed" in out:
        out = out[out["filed"] <= date]
    if "end" in out:
        out = out[out["end"] <= date]
    return out


def debt_asof(debt_frames: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    def val(tag: str) -> float:
        return sc.latest_instant(asof_records(debt_frames.get(tag, pd.DataFrame()), date))
    lease_cur, lease_non = val("LongTermDebtAndFinanceLeaseObligationsCurrent"), val("LongTermDebtAndFinanceLeaseObligationsNoncurrent")
    if np.isfinite(lease_cur) or np.isfinite(lease_non):
        return float(np.nansum([lease_cur, lease_non]))
    cur, non, short = val("LongTermDebtCurrent"), val("LongTermDebtNoncurrent"), val("ShortTermBorrowings")
    if np.isfinite(cur) or np.isfinite(non) or np.isfinite(short):
        return float(np.nansum([cur, non, short]))
    return val("DebtCurrent")


def point_in_time_fundamentals(universe: pd.DataFrame, dates: list[pd.Timestamp], cache: sc.Cache) -> pd.DataFrame:
    cik_map = sc.ticker_cik_map(cache, 87600)
    rows = []
    total = len(universe)
    for i, u in universe.iterrows():
        symbol = str(u["Symbol"])
        cik = cik_map.get(symbol.upper())
        if cik is None:
            continue
        facts = sc.fetch_company_facts(cik, cache, 87600)
        if not facts:
            continue
        duration = {k: sc.records_for_alternatives(facts, tags) for k, tags in sc.DURATION_TAGS.items()}
        instant = {k: sc.records_for_alternatives(facts, tags) for k, tags in sc.INSTANT_TAGS.items()}
        debt_frames = {tag: sc._records_for_tag(facts, "us-gaap", tag) for tag in [
            "LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "LongTermDebtCurrent", "LongTermDebtNoncurrent", "ShortTermBorrowings", "DebtCurrent"
        ]}
        for date in dates:
            rec = {"date": date, "Symbol": symbol}
            for key, df in duration.items():
                cur, prev = sc.ttm_from_records(asof_records(df, date))
                rec[f"{key}_ttm"], rec[f"{key}_prior_ttm"] = cur, prev
            for key, df in instant.items():
                rec[key] = sc.latest_instant(asof_records(df, date))
            rec["debt"] = debt_asof(debt_frames, date)
            rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"SEC point-in-time fundamentals: {i + 1}/{total}", flush=True)
    return pd.DataFrame(rows)


def download_range(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, volumes = [], []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        data = yf.download(batch, start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                           interval="1d", auto_adjust=True, progress=False, group_by="column", threads=True)
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            closes.append(data["Close"] if "Close" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index))
            volumes.append(data["Volume"] if "Volume" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index))
        else:
            closes.append(pd.DataFrame({batch[0]: data["Close"]}, index=data.index))
            volumes.append(pd.DataFrame({batch[0]: data.get("Volume")}, index=data.index))
    if not closes:
        raise RuntimeError("Price download returned no data")
    close = pd.concat(closes, axis=1).sort_index().loc[:, lambda x: ~x.columns.duplicated()]
    volume = pd.concat(volumes, axis=1).sort_index().loc[:, lambda x: ~x.columns.duplicated()]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    return close, volume


def percentile_last(df: pd.DataFrame, col: str, years: int = 10) -> float:
    if df.empty:
        return np.nan
    cutoff = df["date"].max() - pd.DateOffset(years=years)
    s = df.loc[df["date"] >= cutoff, col].dropna()
    if len(s) < 20:
        s = df[col].dropna()
    return float(s.rank(pct=True).iloc[-1]) if not s.empty else np.nan


def macro_asof(history: dict[str, pd.DataFrame], date: pd.Timestamp) -> sc.MacroState:
    data = {}
    rows = []
    for name, series_id in sc.FRED_SERIES.items():
        cutoff = date - pd.Timedelta(days=MACRO_LAG_DAYS.get(name, 1))
        df = history[name]
        df = df[df["date"] <= cutoff].copy()
        data[name] = df
        value = float(df[name].dropna().iloc[-1]) if not df.empty and df[name].notna().any() else np.nan
        rows.append({"name": name, "series": series_id, "date": df["date"].iloc[-1] if not df.empty else pd.NaT,
                     "value": value, "pct10y": percentile_last(df, name)})
    snap = pd.DataFrame(rows)
    vals = snap.set_index("name")["value"].to_dict()
    pcts = snap.set_index("name")["pct10y"].to_dict()
    cpi = data["cpi"]
    cpi_yoy = np.nan
    if len(cpi) >= 13:
        s = cpi.set_index("date")["cpi"].sort_index()
        cpi_yoy = float(s.iloc[-1] / s.iloc[-13] - 1) * 100
    sahm, curve, hy, nfci = vals.get("sahm", np.nan), vals.get("curve_10y3m", np.nan), vals.get("hy_oas", np.nan), vals.get("nfci", np.nan)
    real10, cfnai = vals.get("real10y", np.nan), vals.get("cfnai", np.nan)
    recession = np.nanmean([
        min(max((sahm if np.isfinite(sahm) else 0) / 0.5, 0), 1),
        pcts.get("hy_oas", np.nan), pcts.get("nfci", np.nan),
        1.0 if np.isfinite(curve) and curve < 0 else 0.0,
        1 - pcts.get("cfnai", np.nan) if np.isfinite(pcts.get("cfnai", np.nan)) else np.nan,
    ])
    inflation = np.nanmean([min(max((cpi_yoy - 2.0) / 3.0, 0), 1) if np.isfinite(cpi_yoy) else np.nan, pcts.get("real10y", np.nan)])
    liquidity = np.nanmean([pcts.get("hy_oas", np.nan), pcts.get("nfci", np.nan)])
    rate_pressure = np.nanmean([pcts.get("real10y", np.nan), pcts.get("fedfunds", np.nan)])
    if np.isfinite(sahm) and sahm >= 0.5:
        regime = "RECESSION"
    elif np.isfinite(recession) and recession >= 0.72:
        regime = "RECESSION"
    elif np.isfinite(cpi_yoy) and cpi_yoy >= 3.0 and np.isfinite(cfnai) and cfnai < 0 and np.isfinite(real10) and real10 >= 1.5:
        regime = "STAGFLATION"
    elif np.isfinite(real10) and real10 >= 1.5 and np.isfinite(rate_pressure) and rate_pressure >= 0.60:
        regime = "HIGH_REAL_RATES"
    elif (not np.isfinite(sahm) or sahm < 0.3) and (not np.isfinite(hy) or hy < 4.5) and (not np.isfinite(nfci) or nfci < 0) and (not np.isfinite(curve) or curve > 0):
        regime = "RISK_ON"
    else:
        regime = "NEUTRAL"
    return sc.MacroState(regime, recession, inflation, liquidity, rate_pressure, snap)


def month_ends(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    idx = index[(index >= start) & (index <= end)]
    if idx.empty:
        return []
    frame = pd.DataFrame({"date": idx, "month": idx.to_period("M")})
    return frame.groupby("month")["date"].max().tolist()


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 12
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * math.sqrt(12))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    return {"total_return": total, "cagr": cagr, "annual_volatility": vol, "sharpe_0rf": sharpe,
            "sortino_0rf": float(r.mean() / downside * math.sqrt(12)) if np.isfinite(downside) and downside > 0 else np.nan,
            "max_drawdown": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "positive_month_rate": float((r > 0).mean()), "months": len(r)}


def run(args) -> None:
    config = sc.load_config(args.config)
    cache = sc.Cache(Path(config.get("cache_dir", ".cache_sp500_screener")))
    universe = sc.get_universe(cache, 87600, None)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc).date())
    symbols = universe["Yahoo Symbol"].tolist() + ["SPY"] + list(config["sector_etfs"].values())
    close, volume = download_range(list(dict.fromkeys(symbols)), start - pd.Timedelta(days=430), end, int(config.get("price_batch_size", 80)))
    dates = month_ends(close.index, start, end)
    if len(dates) < 13:
        raise RuntimeError("Need at least 13 monthly observations")
    signal_dates = dates[:-1]
    print(f"Backtest: {signal_dates[0].date()} to {dates[-1].date()}, {len(signal_dates)} holding periods", flush=True)
    fred = {name: sc.fred_series(cache, name, sid, 87600) for name, sid in sc.FRED_SERIES.items()}
    fund_panel = point_in_time_fundamentals(universe, signal_dates, cache)
    fund_panel["date"] = pd.to_datetime(fund_panel["date"])
    yahoo = dict(zip(universe["Symbol"], universe["Yahoo Symbol"]))
    cost = args.cost_bps / 10000
    prev_end = {}
    perf, holdings = [], []
    for i, date in enumerate(signal_dates):
        next_date = dates[i + 1]
        tech = sc.price_features(close.loc[:date], volume.loc[:date], universe)
        sector = sc.sector_rotation(close.loc[:date], universe, tech, config["sector_etfs"])
        fund = fund_panel[fund_panel["date"] == date].drop(columns="date")
        macro = macro_asof(fred, date)
        ranked = sc.score_stocks(sc.build_features(universe, tech, fund, sector), macro, config)
        port = sc.build_portfolio(ranked, args.portfolio_n, args.max_stock_weight, args.max_sector_weight)
        if port.empty:
            continue
        target = dict(zip(port["Symbol"], port["weight"]))
        turnover = 1.0 if not prev_end else 0.5 * sum(abs(target.get(s, 0) - prev_end.get(s, 0)) for s in set(target) | set(prev_end))
        indiv = {}
        for sym in target:
            ys = yahoo.get(sym)
            if ys not in close:
                continue
            s = close[ys].loc[:next_date].dropna()
            p0s = s[s.index <= date]
            if p0s.empty or s.empty:
                continue
            p0, p1 = float(p0s.iloc[-1]), float(s.iloc[-1])
            if p0 > 0:
                indiv[sym] = p1 / p0 - 1
        valid_w = sum(target[s] for s in indiv)
        if valid_w <= 0:
            continue
        gross = sum(target[s] * indiv[s] for s in indiv) / valid_w
        net = gross - turnover * cost
        spy0 = float(close["SPY"].loc[:date].dropna().iloc[-1])
        spy1 = float(close["SPY"].loc[:next_date].dropna().iloc[-1])
        spy_ret = spy1 / spy0 - 1
        raw_end = {s: target[s] * (1 + indiv.get(s, 0)) for s in target}
        den = sum(raw_end.values())
        prev_end = {s: v / den for s, v in raw_end.items()} if den > 0 else target
        perf.append({"date": date, "next_date": next_date, "strategy_return": net, "gross_return": gross, "spy_return": spy_ret,
                     "turnover": turnover, "cost": turnover * cost, "regime": macro.regime, "holdings": len(port)})
        for _, h in port.iterrows():
            holdings.append({"date": date, "Symbol": h["Symbol"], "Security": h["Security"], "GICS Sector": h["GICS Sector"], "score": h["score"], "weight": h["weight"]})
        print(f"{date.date()} {macro.regime:16s} strategy={net:+.2%} SPY={spy_ret:+.2%} turnover={turnover:.1%}", flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    perf = pd.DataFrame(perf)
    if perf.empty:
        raise RuntimeError("No valid backtest periods")
    perf["strategy_equity"] = (1 + perf["strategy_return"]).cumprod()
    perf["spy_equity"] = (1 + perf["spy_return"]).cumprod()
    s_stats, b_stats = perf_stats(perf["strategy_return"]), perf_stats(perf["spy_return"])
    active = perf["strategy_return"] - perf["spy_return"]
    s_stats["avg_monthly_turnover"] = float(perf["turnover"].mean())
    s_stats["annualized_excess_return_arith"] = float(active.mean() * 12)
    s_stats["information_ratio"] = float(active.mean() / active.std(ddof=1) * math.sqrt(12)) if active.std(ddof=1) > 0 else np.nan
    summary = pd.DataFrame([{"portfolio": "regime_factor", **s_stats}, {"portfolio": "SPY", **b_stats}])
    summary.to_csv(out / "summary.csv", index=False)
    perf.to_csv(out / "equity_curve.csv", index=False)
    pd.DataFrame(holdings).to_csv(out / "holdings.csv", index=False)
    pd.DataFrame([
        ["universe", "Current S&P 500 constituents; survivorship bias remains"],
        ["rebalance", "Monthly, signal at month-end close; next month return"],
        ["fundamental_timing", "Only SEC facts with filed/end date <= signal date"],
        ["macro_timing", "Conservative publication lags applied"],
        ["transaction_cost", f"{args.cost_bps} bps one-way * turnover"],
        ["known_bias", "FRED latest-vintage history introduces macro revision bias"],
    ], columns=["assumption", "value"]).to_csv(out / "assumptions.csv", index=False)
    print("\n=== BACKTEST SUMMARY ===\n" + summary.to_string(index=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--portfolio-n", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--max-stock-weight", type=float, default=0.12)
    p.add_argument("--max-sector-weight", type=float, default=0.30)
    p.add_argument("--output-dir", default="data/backtest")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
