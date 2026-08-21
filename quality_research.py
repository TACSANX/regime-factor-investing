from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import backtest_v2 as v2
import screener as sc


QUALITY_PROXY_VERSION = "qmj_proxy_v1"
_REQUIRED_PANEL_COLUMNS = {
    "gross_profit_ttm",
    "gross_profit_prior_ttm",
    "assets_prior",
    "equity_prior",
}

# QMJ profitability uses gross profits explicitly. Keep this research-only
# extension out of the live screener until the hypothesis survives validation.
sc.DURATION_TAGS.setdefault(
    "gross_profit",
    [("us-gaap", "GrossProfit")],
)

_original_snapshot_for_cutoff = v2._snapshot_for_cutoff
_original_point_in_time_panel = v2.point_in_time_fundamentals_fast
_original_price_features = sc.price_features
_original_build_features = sc.build_features
_original_score_stocks = sc.score_stocks


def latest_and_prior_instant(df: pd.DataFrame) -> tuple[float, float]:
    """Return the latest instant fact and the closest fact roughly one year earlier."""
    if df.empty or "val" not in df or "end" not in df:
        return np.nan, np.nan
    x = df.dropna(subset=["end", "val"]).copy()
    if "form" in x:
        filings = x[x["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A", "8-K"])]
        if not filings.empty:
            x = filings
    if x.empty:
        return np.nan, np.nan

    sort_cols = ["end"] + (["filed"] if "filed" in x else [])
    x = x.sort_values(sort_cols).drop_duplicates("end", keep="last")
    latest = x.iloc[-1]
    latest_end = pd.Timestamp(latest["end"])
    prior = x[x["end"] < latest_end].copy()
    if prior.empty:
        return float(latest["val"]), np.nan

    prior["distance_days"] = (latest_end - prior["end"]).dt.days
    year_like = prior[prior["distance_days"].between(240, 500)].copy()
    if not year_like.empty:
        year_like["distance_from_year"] = (year_like["distance_days"] - 365).abs()
        prior_row = year_like.sort_values(["distance_from_year", "end"]).iloc[0]
    else:
        prior_row = prior.iloc[-1]
    return float(latest["val"]), float(prior_row["val"])


def _snapshot_for_cutoff_qmj(
    symbol: str,
    cutoff: pd.Timestamp,
    duration: dict,
    instant: dict,
    debt_frames: dict,
) -> dict:
    rec = _original_snapshot_for_cutoff(symbol, cutoff, duration, instant, debt_frames)
    for key, frame in instant.items():
        _, prior = latest_and_prior_instant(bt.asof_records(frame, cutoff))
        rec[f"{key}_prior"] = prior
    return rec


def point_in_time_fundamentals_fast_qmj(
    universe: pd.DataFrame,
    dates: list[pd.Timestamp],
    cache: sc.Cache,
) -> pd.DataFrame:
    """Reuse V2 cache, rebuilding once if it predates QMJ proxy fields."""
    panel = _original_point_in_time_panel(universe, dates, cache)
    if _REQUIRED_PANEL_COLUMNS.issubset(panel.columns):
        return panel

    # Old Actions caches can contain a valid V2 panel that lacks the newly
    # requested SEC facts. Remove only the derived panel; raw SEC JSON remains.
    for path in cache.root.glob("fund_panel_v2_*.pkl.gz"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    panel = _original_point_in_time_panel(universe, dates, cache)
    missing = _REQUIRED_PANEL_COLUMNS - set(panel.columns)
    if missing:
        raise RuntimeError(f"QMJ research panel missing columns: {sorted(missing)}")
    return panel


def price_features_qmj(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    base = _original_price_features(close, volume, universe)
    if base.empty:
        return base
    if "SPY" not in close:
        base["beta_252"] = np.nan
        base["idio_vol_252"] = np.nan
        return base

    spy_ret = close["SPY"].pct_change()
    yahoo = dict(zip(universe["Symbol"].astype(str), universe["Yahoo Symbol"].astype(str)))
    rows = []
    for symbol in base["Symbol"].astype(str):
        ys = yahoo.get(symbol)
        beta = np.nan
        idio = np.nan
        if ys in close:
            pair = pd.concat(
                [close[ys].pct_change().rename("stock"), spy_ret.rename("market")],
                axis=1,
            ).dropna().tail(252)
            if len(pair) >= 126:
                market_var = float(pair["market"].var(ddof=1))
                if np.isfinite(market_var) and market_var > 0:
                    beta = float(pair["stock"].cov(pair["market"]) / market_var)
                    resid = (
                        pair["stock"] - pair["stock"].mean()
                        - beta * (pair["market"] - pair["market"].mean())
                    )
                    idio = float(resid.std(ddof=1) * np.sqrt(252))
        rows.append({"Symbol": symbol, "beta_252": beta, "idio_vol_252": idio})
    return base.merge(pd.DataFrame(rows), on="Symbol", how="left")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _ratio(num: pd.Series, den: pd.Series, positive_denominator: bool = False) -> pd.Series:
    d = pd.to_numeric(den, errors="coerce").copy()
    if positive_denominator:
        d = d.where(d > 0)
    else:
        d = d.where(d.abs() > 1e-12)
    return pd.to_numeric(num, errors="coerce") / d


def add_qmj_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a point-in-time QMJ-inspired quality proxy.

    This is not an exact QMJ replication. It uses SEC Company Facts and the
    available market history:
      * profitability: GPOA, ROE, ROA, CFOA, gross margin, low accruals;
      * growth: one-year changes in five profitability measures;
      * safety: low beta, low idiosyncratic volatility, low leverage.

    Published QMJ uses five-year residual per-share growth and additional
    bankruptcy/earnings-volatility safety measures, which are not reproduced.
    """
    out = df.copy()
    assets = _col(out, "assets")
    assets_prior = _col(out, "assets_prior")
    equity = _col(out, "equity")
    equity_prior = _col(out, "equity_prior")
    revenue = _col(out, "revenue_ttm")
    revenue_prior = _col(out, "revenue_prior_ttm")
    gross_profit = _col(out, "gross_profit_ttm")
    gross_profit_prior = _col(out, "gross_profit_prior_ttm")
    net_income = _col(out, "net_income_ttm")
    net_income_prior = _col(out, "net_income_prior_ttm")
    cfo = _col(out, "cfo_ttm")
    cfo_prior = _col(out, "cfo_prior_ttm")
    debt = _col(out, "debt")

    out["qmj_gpoa"] = _ratio(gross_profit, assets, True)
    out["qmj_roe"] = _ratio(net_income, equity, True)
    out["qmj_roa"] = _ratio(net_income, assets, True)
    out["qmj_cfoa"] = _ratio(cfo, assets, True)
    out["qmj_gross_margin"] = _ratio(gross_profit, revenue, True)
    # Cash earnings above accounting earnings is a practical SEC-CFO proxy for
    # low accruals, not the paper's exact working-capital formula.
    out["qmj_low_accrual"] = _ratio(cfo - net_income, assets, True)

    out["qmj_growth_gpoa_1y"] = _ratio(gross_profit - gross_profit_prior, assets_prior, True)
    out["qmj_growth_roe_1y"] = _ratio(net_income - net_income_prior, equity_prior, True)
    out["qmj_growth_roa_1y"] = _ratio(net_income - net_income_prior, assets_prior, True)
    out["qmj_growth_cfoa_1y"] = _ratio(cfo - cfo_prior, assets_prior, True)
    prior_margin = _ratio(gross_profit_prior, revenue_prior, True)
    out["qmj_growth_gross_margin_1y"] = out["qmj_gross_margin"] - prior_margin

    out["qmj_leverage"] = _ratio(debt, assets, True)

    score_specs = {
        "qmj_gpoa": True,
        "qmj_roe": True,
        "qmj_roa": True,
        "qmj_cfoa": True,
        "qmj_gross_margin": True,
        "qmj_low_accrual": True,
        "qmj_growth_gpoa_1y": True,
        "qmj_growth_roe_1y": True,
        "qmj_growth_roa_1y": True,
        "qmj_growth_cfoa_1y": True,
        "qmj_growth_gross_margin_1y": True,
        "beta_252": False,
        "idio_vol_252": False,
        "qmj_leverage": False,
    }
    for col, higher_is_better in score_specs.items():
        out[f"{col}_score"] = sc.blended_percentile(out, col, higher_is_better)

    out["qmj_profitability"] = sc.weighted_available(
        out,
        [
            ("qmj_gpoa_score", 1.0),
            ("qmj_roe_score", 1.0),
            ("qmj_roa_score", 1.0),
            ("qmj_cfoa_score", 1.0),
            ("qmj_gross_margin_score", 1.0),
            ("qmj_low_accrual_score", 1.0),
        ],
    )
    out["qmj_growth_proxy"] = sc.weighted_available(
        out,
        [
            ("qmj_growth_gpoa_1y_score", 1.0),
            ("qmj_growth_roe_1y_score", 1.0),
            ("qmj_growth_roa_1y_score", 1.0),
            ("qmj_growth_cfoa_1y_score", 1.0),
            ("qmj_growth_gross_margin_1y_score", 1.0),
        ],
    )
    out["qmj_safety"] = sc.weighted_available(
        out,
        [
            ("beta_252_score", 1.0),
            ("idio_vol_252_score", 1.0),
            ("qmj_leverage_score", 1.0),
        ],
    )

    components = out[["qmj_profitability", "qmj_growth_proxy", "qmj_safety"]]
    out["quality_qmj_proxy"] = sc.weighted_available(
        out,
        [
            ("qmj_profitability", 1.0),
            ("qmj_growth_proxy", 1.0),
            ("qmj_safety", 1.0),
        ],
    )
    out["quality_qmj_component_count"] = components.notna().sum(axis=1)
    out.loc[out["quality_qmj_component_count"] < 2, "quality_qmj_proxy"] = np.nan
    return out


def build_features_qmj(
    universe: pd.DataFrame,
    tech: pd.DataFrame,
    fund: pd.DataFrame,
    sector: pd.DataFrame,
) -> pd.DataFrame:
    return add_qmj_proxy_features(_original_build_features(universe, tech, fund, sector))


def score_stocks_qmj(df: pd.DataFrame, macro: sc.MacroState, config: dict) -> pd.DataFrame:
    quality_col = config.get("_research_quality_column")
    if not quality_col:
        return _original_score_stocks(df, macro, config)
    if quality_col not in df:
        raise KeyError(f"Research quality column not found: {quality_col}")

    x = df.copy()
    x["quality"] = x[quality_col]
    factors = ["momentum", "quality", "value", "growth", "low_vol", "sector_rotation"]
    x["data_completeness"] = x[factors].notna().mean(axis=1)
    x["fundamental_completeness"] = x[["quality", "value", "growth"]].notna().mean(axis=1)
    return _original_score_stocks(x, macro, config)


# Patch only the research process importing this module. The live daily screener
# never imports quality_research.py.
v2._snapshot_for_cutoff = _snapshot_for_cutoff_qmj
v2.point_in_time_fundamentals_fast = point_in_time_fundamentals_fast_qmj
sc.price_features = price_features_qmj
sc.build_features = build_features_qmj
sc.score_stocks = score_stocks_qmj
