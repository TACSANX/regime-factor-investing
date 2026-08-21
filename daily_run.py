from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import screener as sc


def portfolio_diagnostics(portfolio: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame([{
            "holdings": 0,
            "max_stock_weight": np.nan,
            "top3_stock_weight": np.nan,
            "stock_hhi": np.nan,
            "effective_stock_count": np.nan,
            "max_sector_weight": np.nan,
            "sector_hhi": np.nan,
            "effective_sector_count": np.nan,
        }])

    w = portfolio["weight"].astype(float)
    sector_w = portfolio.groupby("GICS Sector")["weight"].sum().astype(float)
    stock_hhi = float(np.square(w).sum())
    sector_hhi = float(np.square(sector_w).sum())
    return pd.DataFrame([{
        "holdings": int(len(portfolio)),
        "max_stock_weight": float(w.max()),
        "top3_stock_weight": float(w.nlargest(min(3, len(w))).sum()),
        "stock_hhi": stock_hhi,
        "effective_stock_count": 1.0 / stock_hhi if stock_hhi > 0 else np.nan,
        "max_sector_weight": float(sector_w.max()),
        "sector_hhi": sector_hhi,
        "effective_sector_count": 1.0 / sector_hhi if sector_hhi > 0 else np.nan,
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily uncapped S&P 500 regime-factor screener")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--universe-csv", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--portfolio", type=int, default=None)
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    config = sc.load_config(args.config)
    cache = sc.Cache(Path(config.get("cache_dir", ".cache_sp500_screener")))
    hours = config.get("cache_hours", {})
    universe = sc.get_universe(cache, float(hours.get("universe", 12)), args.universe_csv)
    symbols = universe["Yahoo Symbol"].tolist() + ["SPY"] + list(config["sector_etfs"].values())
    close, volume = sc.download_prices(symbols, config.get("price_period", "2y"), int(config.get("price_batch_size", 80)))
    if close.empty:
        raise RuntimeError("Price download returned no data.")

    tech = sc.price_features(close, volume, universe)
    sector = sc.sector_rotation(close, universe, tech, config["sector_etfs"])
    fund = sc.fetch_fundamentals(universe, cache, float(hours.get("sec", 24)))
    macro = sc.macro_state(cache, float(hours.get("macro", 6)))
    ranked = sc.score_stocks(sc.build_features(universe, tech, fund, sector), macro, config)

    # config.yaml intentionally uses non-binding 100% caps. The original
    # allocator therefore behaves as top-N inverse-volatility x score weighting
    # without a hard stock or sector concentration constraint.
    portfolio = sc.build_portfolio(
        ranked,
        args.portfolio or int(config.get("portfolio_n", 10)),
        float(config.get("max_stock_weight", 1.0)),
        float(config.get("max_sector_weight", 1.0)),
    )

    ranked.to_csv(args.output, index=False)
    portfolio.to_csv("portfolio.csv", index=False)
    macro.snapshot.to_csv("macro_snapshot.csv", index=False)
    diagnostics = portfolio_diagnostics(portfolio)
    diagnostics.insert(0, "macro_regime", macro.regime)
    diagnostics.to_csv("portfolio_diagnostics.csv", index=False)

    sc.print_summary(ranked, portfolio, macro, args.top or int(config.get("top_n", 20)))
    print("\n=== CONCENTRATION DIAGNOSTICS ===")
    print(diagnostics.round(4).to_string(index=False))
    print(f"\nSaved: {args.output}, portfolio.csv, macro_snapshot.csv, portfolio_diagnostics.csv")


if __name__ == "__main__":
    main()
