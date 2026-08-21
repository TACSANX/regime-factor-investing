from __future__ import annotations

import pandas as pd

import forward_shadow as fs


def strategy_entries(registry: dict, specs: dict[str, dict]) -> list[dict]:
    out = []
    for raw in registry["strategies"]:
        if isinstance(raw, str):
            name = raw
            source = raw
            allocation = None
        else:
            name = str(raw["name"])
            source = str(raw["source_strategy"])
            allocation = raw.get("allocation_override")
        if source not in specs:
            raise RuntimeError(f"Frozen source strategy missing from model: {source}")
        spec = dict(specs[source])
        if allocation is not None:
            spec["allocation"] = str(allocation)
        out.append({"name": name, "source": source, "spec": spec})
    return out


def append_targets_r2(
    signals: pd.DataFrame,
    signal_date: pd.Timestamp,
    universe: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    registry: dict,
    model_root,
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
            & (pd.to_datetime(signals["signal_date"], errors="coerce") == signal_date)
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
    entries = strategy_entries(registry, specs)
    score_names = sorted({e["spec"]["score_variant"] for e in entries})
    ranked = {name: sc.score_stocks(features, macro, configs[name]) for name in score_names}
    yahoo = dict(zip(universe["Symbol"], universe["Yahoo Symbol"]))

    rows = []
    for entry in entries:
        label = entry["name"]
        spec = entry["spec"]
        port = rb._portfolio(ranked[spec["score_variant"]], spec)
        if port.empty:
            raise RuntimeError(f"Frozen strategy produced empty portfolio: {label}")
        for _, h in port.iterrows():
            rows.append({
                "cohort": cohort,
                "frozen_commit": str(registry["frozen_commit"]),
                "signal_date": signal_date.date().isoformat(),
                "strategy": label,
                "regime": macro.regime,
                "Symbol": h["Symbol"],
                "Yahoo Symbol": yahoo.get(h["Symbol"], h["Symbol"]),
                "Security": h["Security"],
                "GICS Sector": h["GICS Sector"],
                "score": float(h["score"]),
                "weight": float(h["weight"]),
                "execution_date": "",
                "traded_notional": float("nan"),
                "cost_bps_per_dollar_traded": float(registry["transaction_cost_bps_per_dollar_traded"]),
            })
    new = pd.DataFrame(rows, columns=fs.SIGNAL_COLUMNS)
    print(f"Registered {len(entries)} r2 strategies for {signal_date.date()} regime={macro.regime}", flush=True)
    return pd.concat([signals, new], ignore_index=True)


fs.append_targets = append_targets_r2


if __name__ == "__main__":
    fs.main()
