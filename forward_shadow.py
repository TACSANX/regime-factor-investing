from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


def load_frozen_modules(model_root: Path):
    sys.path.insert(0, str(model_root.resolve()))
    import screener as sc  # type: ignore
    import backtest as bt  # type: ignore
    import backtest_v2 as v2  # type: ignore
    import research_backtest as rb  # type: ignore
    import research_candidates as rc  # type: ignore
    return sc, bt, v2, rb, rc


def load_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path)


def target_period(now_jst: datetime) -> pd.Period:
    return pd.Period(pd.Timestamp(now_jst.date()), freq="M") - 1


def last_session_in_period(close: pd.DataFrame, symbol: str, period: pd.Period) -> pd.Timestamp:
    if symbol not in close.columns:
        raise RuntimeError(f"Missing benchmark price series: {symbol}")
    s = close[symbol].dropna()
    idx = s.index[pd.PeriodIndex(s.index, freq="M") == period]
    if len(idx) == 0:
        raise RuntimeError(f"No {symbol} session in target period {period}")
    return pd.Timestamp(idx[-1]).tz_localize(None)


def end_weights(target: dict[str, float], returns: dict[str, float]) -> dict[str, float]:
    raw = {s: target[s] * (1.0 + returns.get(s, 0.0)) for s in target}
    den = float(sum(raw.values()))
    return {s: v / den for s, v in raw.items()} if den > 0 else dict(target)


def traded_notional(target: dict[str, float], previous_end: dict[str, float] | None) -> float:
    if not previous_end:
        return 1.0
    half_l1 = 0.5 * sum(
        abs(target.get(s, 0.0) - previous_end.get(s, 0.0))
        for s in set(target) | set(previous_end)
    )
    return float(2.0 * half_l1)


def price_return(v2, open_px: pd.DataFrame, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float(v2.price_return_at_open(open_px, symbol, start, end))


def append_targets(
    signals: pd.DataFrame,
    signal_date: pd.Timestamp,
    universe: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    registry: dict,
    model_root: Path,
    sc,
    bt,
    v2,
    rb,
    rc,
    cache,
) -> pd.DataFrame:
    cohort = str(registry["cohort"])
    if not signals.empty:
        existing = signals[
            (signals["cohort"].astype(str) == cohort)
            & (pd.to_datetime(signals["signal_date"]) == signal_date)
        ]
        if not existing.empty:
            print(f"Targets already registered for {signal_date.date()} ({cohort})", flush=True)
            return signals

    config = sc.load_config(str(model_root / "config.yaml"))
    tech = sc.price_features(close.loc[:signal_date], volume.loc[:signal_date], universe)
    sector = sc.sector_rotation(close.loc[:signal_date], universe, tech, config["sector_etfs"])
    fund_panel = v2.point_in_time_fundamentals_fast(universe, [signal_date], cache)
    fund_panel["date"] = pd.to_datetime(fund_panel["date"])
    fund = fund_panel[fund_panel["date"] == signal_date].drop(columns="date")
    features = sc.build_features(universe, tech, fund, sector)

    fred = {
        name: sc.fred_series(cache, name, sid, float(config.get("cache_hours", {}).get("macro", 6)))
        for name, sid in sc.FRED_SERIES.items()
    }
    bt.MACRO_LAG_DAYS = v2.MACRO_LAG_DAYS_V2.copy()
    macro = bt.macro_asof(fred, signal_date)

    configs = rc.score_variants(config)
    specs = {s["strategy"]: s for s in rc.strategy_specs()}
    wanted = [str(x) for x in registry["strategies"]]
    missing = [x for x in wanted if x not in specs]
    if missing:
        raise RuntimeError(f"Frozen registry strategies missing from frozen model: {missing}")

    score_names = sorted({specs[s]["score_variant"] for s in wanted})
    ranked = {
        name: sc.score_stocks(features, macro, configs[name])
        for name in score_names
    }
    yahoo = dict(zip(universe["Symbol"], universe["Yahoo Symbol"]))
    rows = []
    for strategy in wanted:
        spec = specs[strategy]
        port = rb._portfolio(ranked[spec["score_variant"]], spec)
        if port.empty:
            raise RuntimeError(f"Frozen strategy produced empty portfolio: {strategy}")
        for _, h in port.iterrows():
            rows.append({
                "cohort": cohort,
                "frozen_commit": str(registry["frozen_commit"]),
                "signal_date": signal_date.date().isoformat(),
                "strategy": strategy,
                "regime": macro.regime,
                "Symbol": h["Symbol"],
                "Yahoo Symbol": yahoo.get(h["Symbol"], h["Symbol"]),
                "Security": h["Security"],
                "GICS Sector": h["GICS Sector"],
                "score": float(h["score"]),
                "weight": float(h["weight"]),
                "execution_date": "",
                "traded_notional": np.nan,
                "cost_bps_per_dollar_traded": float(registry["transaction_cost_bps_per_dollar_traded"]),
            })
    new = pd.DataFrame(rows)
    print(
        f"Registered {len(wanted)} frozen strategies for signal {signal_date.date()} regime={macro.regime}",
        flush=True,
    )
    return pd.concat([signals, new], ignore_index=True)


def finalize_executions(signals: pd.DataFrame, open_px: pd.DataFrame, v2) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    signal_dates = sorted(out["signal_date"].dropna().unique())

    for i, sd_raw in enumerate(signal_dates):
        sd = pd.Timestamp(sd_raw)
        exec_date = v2.next_session(open_px.index, sd)
        if exec_date is None:
            continue
        exec_date = pd.Timestamp(exec_date)
        current_mask = out["signal_date"].eq(sd)
        for strategy in out.loc[current_mask, "strategy"].drop_duplicates():
            mask = current_mask & out["strategy"].eq(strategy)
            if out.loc[mask, "execution_date"].astype(str).str.len().gt(0).all() and out.loc[mask, "traded_notional"].notna().all():
                continue
            target = dict(zip(out.loc[mask, "Yahoo Symbol"], out.loc[mask, "weight"].astype(float)))
            prev_end = None
            if i > 0:
                prev_sd = pd.Timestamp(signal_dates[i - 1])
                pmask = out["signal_date"].eq(prev_sd) & out["strategy"].eq(strategy)
                if pmask.any():
                    prev_exec_values = out.loc[pmask, "execution_date"].astype(str)
                    if prev_exec_values.str.len().gt(0).all():
                        prev_exec = pd.Timestamp(prev_exec_values.iloc[0])
                        prev_target = dict(zip(out.loc[pmask, "Yahoo Symbol"], out.loc[pmask, "weight"].astype(float)))
                        rets = {}
                        for symbol in prev_target:
                            r = price_return(v2, open_px, symbol, prev_exec, exec_date)
                            if np.isfinite(r):
                                rets[symbol] = r
                        if sum(prev_target[s] for s in rets) >= 0.95:
                            prev_end = end_weights(prev_target, rets)
            tn = traded_notional(target, prev_end)
            out.loc[mask, "execution_date"] = exec_date.date().isoformat()
            out.loc[mask, "traded_notional"] = tn
            print(f"Finalized {strategy} {sd.date()} exec={exec_date.date()} traded={tn:.3f}x NAV", flush=True)
    return out


def realize_periods(signals: pd.DataFrame, returns: pd.DataFrame, open_px: pd.DataFrame, registry: dict, v2) -> pd.DataFrame:
    if signals.empty:
        return returns
    sig = signals.copy()
    sig["signal_date"] = pd.to_datetime(sig["signal_date"])
    signal_dates = sorted(sig["signal_date"].dropna().unique())
    existing = set()
    if not returns.empty:
        existing = set(zip(returns["strategy"].astype(str), pd.to_datetime(returns["signal_date"]).dt.date.astype(str)))
    rows = []
    cost_rate = float(registry["transaction_cost_bps_per_dollar_traded"]) / 10000.0

    for i in range(len(signal_dates) - 1):
        sd = pd.Timestamp(signal_dates[i])
        nd = pd.Timestamp(signal_dates[i + 1])
        for strategy in sig[sig["signal_date"].eq(sd)]["strategy"].drop_duplicates():
            key = (str(strategy), sd.date().isoformat())
            if key in existing:
                continue
            pmask = sig["signal_date"].eq(sd) & sig["strategy"].eq(strategy)
            nmask = sig["signal_date"].eq(nd) & sig["strategy"].eq(strategy)
            if not nmask.any():
                continue
            pexecs = sig.loc[pmask, "execution_date"].astype(str)
            nexecs = sig.loc[nmask, "execution_date"].astype(str)
            if not pexecs.str.len().gt(0).all() or not nexecs.str.len().gt(0).all():
                continue
            start_exec = pd.Timestamp(pexecs.iloc[0])
            end_exec = pd.Timestamp(nexecs.iloc[0])
            target = dict(zip(sig.loc[pmask, "Yahoo Symbol"], sig.loc[pmask, "weight"].astype(float)))
            indiv = {}
            for symbol in target:
                r = price_return(v2, open_px, symbol, start_exec, end_exec)
                if np.isfinite(r):
                    indiv[symbol] = r
            valid_weight = float(sum(target[s] for s in indiv))
            if valid_weight < 0.95:
                print(f"Cannot realize {strategy} {sd.date()}: valid weight={valid_weight:.1%}", flush=True)
                continue
            gross = float(sum(target[s] * indiv[s] for s in indiv) / valid_weight)
            tn = float(sig.loc[pmask, "traded_notional"].dropna().iloc[0])
            cost = tn * cost_rate
            spy_ret = price_return(v2, open_px, "SPY", start_exec, end_exec)
            rsp_ret = price_return(v2, open_px, "RSP", start_exec, end_exec)
            rows.append({
                "cohort": str(registry["cohort"]),
                "strategy": strategy,
                "signal_date": sd.date().isoformat(),
                "execution_date": start_exec.date().isoformat(),
                "next_signal_date": nd.date().isoformat(),
                "exit_execution_date": end_exec.date().isoformat(),
                "gross_return": gross,
                "net_return": gross - cost,
                "traded_notional": tn,
                "transaction_cost": cost,
                "spy_return": spy_ret,
                "rsp_return": rsp_ret,
                "active_return_vs_spy": gross - cost - spy_ret if np.isfinite(spy_ret) else np.nan,
                "valid_weight_fraction": valid_weight,
            })
            print(f"Realized {strategy} {sd.date()}->{nd.date()}: net={gross - cost:+.2%}", flush=True)
    if rows:
        returns = pd.concat([returns, pd.DataFrame(rows)], ignore_index=True)
    return returns


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", required=True)
    p.add_argument("--registry", default="forward_registry.yaml")
    p.add_argument("--output-dir", default="data/forward_shadow")
    args = p.parse_args()

    model_root = Path(args.model_root)
    registry = yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
    sc, bt, v2, rb, rc = load_frozen_modules(model_root)
    config = sc.load_config(str(model_root / "config.yaml"))
    cache = sc.Cache(Path(config.get("cache_dir", ".cache_sp500_screener")))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    signals_path = outdir / "signals.csv"
    returns_path = outdir / "returns.csv"
    signals = load_csv(signals_path)
    returns = load_csv(returns_path)

    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    period = target_period(now_jst)
    min_start = period.start_time - pd.Timedelta(days=430)
    if not signals.empty:
        old_dates = pd.to_datetime(signals["signal_date"], errors="coerce").dropna()
        if not old_dates.empty:
            min_start = min(min_start, old_dates.min() - pd.Timedelta(days=10))

    universe = sc.get_universe(cache, float(config.get("cache_hours", {}).get("universe", 12)), None)
    prior_yahoo = signals["Yahoo Symbol"].dropna().astype(str).tolist() if "Yahoo Symbol" in signals.columns else []
    symbols = universe["Yahoo Symbol"].tolist() + prior_yahoo + ["SPY", "RSP"] + list(config["sector_etfs"].values())
    symbols = list(dict.fromkeys(symbols))
    end = pd.Timestamp(now_jst.date()) + pd.Timedelta(days=2)
    open_px, close, volume = v2.download_ohlcv(symbols, min_start, end, int(config.get("price_batch_size", 80)))
    if close.empty:
        raise RuntimeError("No market data for forward shadow run")

    signal_date = last_session_in_period(close, "SPY", period)
    signals = append_targets(
        signals, signal_date, universe, close, volume, registry, model_root,
        sc, bt, v2, rb, rc, cache,
    )
    signals = finalize_executions(signals, open_px, v2)
    returns = realize_periods(signals, returns, open_px, registry, v2)

    if not signals.empty:
        signals = signals.sort_values(["signal_date", "strategy", "weight"], ascending=[True, True, False])
        signals["signal_date"] = pd.to_datetime(signals["signal_date"]).dt.date.astype(str)
    if not returns.empty:
        returns = returns.sort_values(["signal_date", "strategy"])
    signals.to_csv(signals_path, index=False)
    returns.to_csv(returns_path, index=False)

    pd.DataFrame([{
        "cohort": registry["cohort"],
        "frozen_commit": registry["frozen_commit"],
        "registered_signal_months": int(pd.to_datetime(signals["signal_date"]).nunique()) if not signals.empty else 0,
        "strategies": int(signals["strategy"].nunique()) if not signals.empty else 0,
        "realized_strategy_months": int(len(returns)) if not returns.empty else 0,
        "last_signal_date": signals["signal_date"].max() if not signals.empty else "",
        "last_realized_signal_date": returns["signal_date"].max() if not returns.empty else "",
        "status": "FORWARD_SHADOW_ONLY",
    }]).to_csv(outdir / "status.csv", index=False)


if __name__ == "__main__":
    main()
