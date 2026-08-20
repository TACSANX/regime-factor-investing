from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

FRED_SERIES = {
    "sahm": "SAHMREALTIME",
    "curve_10y3m": "T10Y3M",
    "hy_oas": "BAMLH0A0HYM2",
    "nfci": "NFCI",
    "real10y": "DFII10",
    "fedfunds": "DFF",
    "unrate": "UNRATE",
    "cpi": "CPIAUCSL",
    "cfnai": "CFNAI",
}

DURATION_TAGS = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    "net_income": [("us-gaap", "NetIncomeLoss")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "capex": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsForAdditionsToPropertyPlantAndEquipment"),
    ],
}

INSTANT_TAGS = {
    "assets": [("us-gaap", "Assets")],
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ],
    "shares": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
}

DEBT_TAGS = [
    ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent"),
    ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"),
    ("us-gaap", "LongTermDebtCurrent"),
    ("us-gaap", "LongTermDebtNoncurrent"),
    ("us-gaap", "ShortTermBorrowings"),
    ("us-gaap", "DebtCurrent"),
]


@dataclass
class MacroState:
    regime: str
    recession_risk: float
    inflation_pressure: float
    liquidity_stress: float
    rate_pressure: float
    snapshot: pd.DataFrame


class Cache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str, suffix: str = ".json") -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.root / f"{safe}{suffix}"

    def fresh(self, path: Path, hours: float) -> bool:
        if not path.exists():
            return False
        now = datetime.now(timezone.utc)
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return now - mtime <= timedelta(hours=hours)

    def load_json(self, key: str, hours: float):
        p = self.path(key)
        if not self.fresh(p, hours):
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_json(self, key: str, obj) -> None:
        self.path(key).write_text(json.dumps(obj), encoding="utf-8")

    def load_csv(self, key: str, hours: float):
        p = self.path(key, ".csv")
        if not self.fresh(p, hours):
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    def save_csv(self, key: str, df: pd.DataFrame) -> None:
        df.to_csv(self.path(key, ".csv"), index=False)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def winsorize(s: pd.Series, q: float = 0.03) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if len(valid) < 10:
        return s
    lo, hi = valid.quantile([q, 1 - q])
    return s.clip(lo, hi)


def percentile_score(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    rank = winsorize(s).rank(pct=True, method="average")
    return rank if higher_is_better else 1.0 - rank


def blended_percentile(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    global_rank = percentile_score(df[col], higher_is_better)
    sector_rank = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby("GICS Sector", dropna=False).groups.items():
        vals = df.loc[idx, col]
        sector_rank.loc[idx] = (
            percentile_score(vals, higher_is_better)
            if vals.notna().sum() >= 4
            else global_rank.loc[idx]
        )
    return 0.5 * global_rank + 0.5 * sector_rank


def weighted_available(df: pd.DataFrame, specs: Sequence[Tuple[str, float]]) -> pd.Series:
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for col, weight in specs:
        if col not in df:
            continue
        mask = df[col].notna()
        num.loc[mask] += df.loc[mask, col] * weight
        den.loc[mask] += weight
    return num.div(den.replace(0, np.nan))


def get_universe(cache: Cache, cache_hours: float, universe_csv: Optional[str]) -> pd.DataFrame:
    if universe_csv:
        df = pd.read_csv(universe_csv)
    else:
        df = cache.load_csv("sp500_universe", cache_hours)
        if df is None or len(df) < 450:
            response = requests.get(WIKI_SP500, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            df = pd.read_html(StringIO(response.text))[0]
            cache.save_csv("sp500_universe", df)
    required = {"Symbol", "Security", "GICS Sector"}
    if missing := required - set(df.columns):
        raise ValueError(f"Universe missing columns: {sorted(missing)}")
    cols = [c for c in ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"] if c in df.columns]
    out = df[cols].copy().drop_duplicates("Symbol")
    out["Yahoo Symbol"] = out["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return out.reset_index(drop=True)


def sec_headers() -> dict:
    ua = os.getenv("SEC_USER_AGENT", "")
    if not ua or "@" not in ua:
        raise RuntimeError("Set SEC_USER_AGENT to an identifiable value such as 'Your Name you@example.com'.")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def ticker_cik_map(cache: Cache, hours: float) -> Dict[str, int]:
    data = cache.load_json("sec_company_tickers", hours)
    if data is None:
        response = requests.get(SEC_TICKERS, headers=sec_headers(), timeout=30)
        response.raise_for_status()
        data = response.json()
        cache.save_json("sec_company_tickers", data)
    return {
        str(v.get("ticker", "")).upper(): int(v["cik_str"])
        for v in data.values()
        if v.get("ticker") and v.get("cik_str") is not None
    }


def fetch_company_facts(cik: int, cache: Cache, hours: float):
    key = f"sec_facts_{cik:010d}"
    data = cache.load_json(key, hours)
    if data is not None:
        return data
    try:
        response = requests.get(SEC_FACTS.format(cik=cik), headers=sec_headers(), timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        cache.save_json(key, data)
        time.sleep(0.13)
        return data
    except requests.RequestException:
        return None


def _records_for_tag(facts: dict, namespace: str, tag: str) -> pd.DataFrame:
    node = facts.get("facts", {}).get(namespace, {}).get(tag)
    if not node:
        return pd.DataFrame()
    units = node.get("units", {})
    records = []
    preferred = ["USD", "shares", "USD/shares", "pure"]
    for unit in preferred + [u for u in units if u not in preferred]:
        if units.get(unit):
            records = [{**r, "unit": unit} for r in units[unit]]
            break
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in ["start", "end", "filed"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "val" in df:
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
    if "start" in df and "end" in df:
        df["days"] = (df["end"] - df["start"]).dt.days
    return df


def records_for_alternatives(facts: dict, alternatives) -> pd.DataFrame:
    for namespace, tag in alternatives:
        df = _records_for_tag(facts, namespace, tag)
        if not df.empty:
            return df
    return pd.DataFrame()


def _fy(value):
    try:
        return None if pd.isna(value) else int(value)
    except Exception:
        return None


def annual_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    if "form" in x:
        x = x[x["form"].isin(["10-K", "10-K/A"])]
    if "days" in x:
        x = x[x["days"].between(300, 400)]
    if "fp" in x and x["fp"].astype(str).eq("FY").any():
        x = x[x["fp"].astype(str).eq("FY")]
    x = x.dropna(subset=["end", "val"])
    if "filed" in x:
        x = x.sort_values(["end", "filed"]).drop_duplicates(["start", "end"], keep="last")
    return x.sort_values("end")


def ytd_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    if "form" in x:
        x = x[x["form"].isin(["10-Q", "10-Q/A"])]
    if "days" in x:
        x = x[x["days"].between(60, 310)]
    if "fp" in x:
        x = x[x["fp"].astype(str).isin(["Q1", "Q2", "Q3"])]
    x = x.dropna(subset=["end", "val"])
    if x.empty:
        return x
    sort_cols = ["end", "days"] + (["filed"] if "filed" in x else [])
    return x.sort_values(sort_cols).drop_duplicates("end", keep="last").sort_values("end")


def ttm_from_records(df: pd.DataFrame) -> Tuple[float, float]:
    anns = annual_records(df)
    if anns.empty:
        return np.nan, np.nan
    latest = anns.iloc[-1]
    latest_val = float(latest["val"])
    prior_ann = float(anns.iloc[-2]["val"]) if len(anns) >= 2 else np.nan
    ytd = ytd_records(df)
    if ytd.empty:
        return latest_val, prior_ann

    curr = ytd[ytd["end"] > latest["end"]]
    latest_fy = _fy(latest.get("fy"))
    if latest_fy is not None and "fy" in curr:
        preferred = curr[curr["fy"].map(_fy) == latest_fy + 1]
        if not preferred.empty:
            curr = preferred
    if curr.empty:
        return latest_val, prior_ann
    curr = curr.iloc[-1]

    prior_ytd = ytd[ytd["end"] < latest["end"]]
    if "fp" in prior_ytd:
        prior_ytd = prior_ytd[prior_ytd["fp"].astype(str) == str(curr.get("fp", ""))]
    if latest_fy is not None and "fy" in prior_ytd:
        preferred = prior_ytd[prior_ytd["fy"].map(_fy) == latest_fy]
        if not preferred.empty:
            prior_ytd = preferred
    if prior_ytd.empty:
        return latest_val, prior_ann
    prior_ytd_row = prior_ytd.iloc[-1]
    current_ttm = latest_val + float(curr["val"]) - float(prior_ytd_row["val"])

    previous_ttm = prior_ann
    if len(anns) >= 2 and np.isfinite(prior_ann):
        prev_ann_row = anns.iloc[-2]
        pp = ytd[ytd["end"] < prev_ann_row["end"]]
        if "fp" in pp:
            pp = pp[pp["fp"].astype(str) == str(curr.get("fp", ""))]
        prev_fy = _fy(prev_ann_row.get("fy"))
        if prev_fy is not None and "fy" in pp:
            preferred = pp[pp["fy"].map(_fy) == prev_fy]
            if not preferred.empty:
                pp = preferred
        if not pp.empty:
            previous_ttm = prior_ann + float(prior_ytd_row["val"]) - float(pp.iloc[-1]["val"])
    return current_ttm, previous_ttm


def latest_instant(df: pd.DataFrame) -> float:
    if df.empty or "val" not in df or "end" not in df:
        return np.nan
    x = df.dropna(subset=["end", "val"]).copy()
    if "form" in x:
        filings = x[x["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A", "8-K"])]
        if not filings.empty:
            x = filings
    if x.empty:
        return np.nan
    sort_cols = ["end"] + (["filed"] if "filed" in x else [])
    return float(x.sort_values(sort_cols).iloc[-1]["val"])


def debt_value(facts: dict) -> float:
    def value(tag: str) -> float:
        return latest_instant(_records_for_tag(facts, "us-gaap", tag))

    lease_current = value("LongTermDebtAndFinanceLeaseObligationsCurrent")
    lease_noncurrent = value("LongTermDebtAndFinanceLeaseObligationsNoncurrent")
    if np.isfinite(lease_current) or np.isfinite(lease_noncurrent):
        return float(np.nansum([lease_current, lease_noncurrent]))

    current = value("LongTermDebtCurrent")
    noncurrent = value("LongTermDebtNoncurrent")
    short = value("ShortTermBorrowings")
    if np.isfinite(current) or np.isfinite(noncurrent) or np.isfinite(short):
        return float(np.nansum([current, noncurrent, short]))

    return value("DebtCurrent")


def fetch_fundamentals(universe: pd.DataFrame, cache: Cache, hours: float) -> pd.DataFrame:
    cik_map = ticker_cik_map(cache, hours)
    rows = []
    for row in universe.itertuples(index=False):
        symbol = str(row.Symbol)
        cik = cik_map.get(symbol.upper())
        if cik is None:
            rows.append({"Symbol": symbol})
            continue
        facts = fetch_company_facts(cik, cache, hours)
        if not facts:
            rows.append({"Symbol": symbol})
            continue
        out = {"Symbol": symbol}
        for key, tags in DURATION_TAGS.items():
            current, previous = ttm_from_records(records_for_alternatives(facts, tags))
            out[f"{key}_ttm"] = current
            out[f"{key}_prior_ttm"] = previous
        for key, tags in INSTANT_TAGS.items():
            out[key] = latest_instant(records_for_alternatives(facts, tags))
        out["debt"] = debt_value(facts)
        rows.append(out)
    return pd.DataFrame(rows)


def download_prices(symbols: Sequence[str], period: str, batch_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    closes, volumes = [], []
    symbols = list(dict.fromkeys(symbols))
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        try:
            data = yf.download(
                batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception:
            continue
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            closes.append(data["Close"].copy() if "Close" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index))
            volumes.append(data["Volume"].copy() if "Volume" in data.columns.get_level_values(0) else pd.DataFrame(index=data.index))
        else:
            sym = batch[0]
            closes.append(pd.DataFrame({sym: data["Close"]}, index=data.index))
            volumes.append(pd.DataFrame({sym: data.get("Volume")}, index=data.index))
    if not closes:
        return pd.DataFrame(), pd.DataFrame()
    close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index()
    return close.loc[:, ~close.columns.duplicated()], volume.loc[:, ~volume.columns.duplicated()]


def _ret(s: pd.Series, days: int, skip: int = 0) -> float:
    x = s.dropna()
    if len(x) <= days + skip:
        return np.nan
    end = x.iloc[-1 - skip] if skip else x.iloc[-1]
    start = x.iloc[-1 - days - skip]
    return np.nan if start == 0 else float(end / start - 1)


def _rsi(s: pd.Series, window: int = 14) -> float:
    delta = s.dropna().diff()
    if len(delta) <= window:
        return np.nan
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def price_features(close: pd.DataFrame, volume: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    spy = close.get("SPY")
    rows = []
    for row in universe.itertuples(index=False):
        ysym = universe.loc[universe["Symbol"].eq(row.Symbol), "Yahoo Symbol"].iloc[0]
        if ysym not in close:
            continue
        s = close[ysym].dropna()
        if len(s) < 126:
            continue
        r = s.pct_change().dropna()
        ma50 = s.tail(50).mean() if len(s) >= 50 else np.nan
        ma200 = s.tail(200).mean() if len(s) >= 200 else np.nan
        rel6 = _ret(s, 126) - _ret(spy, 126) if spy is not None else np.nan
        rel3 = _ret(s, 63) - _ret(spy, 63) if spy is not None else np.nan
        adv = np.nan
        if ysym in volume:
            vv = volume[ysym].reindex(s.index).tail(20)
            if vv.notna().any():
                adv = float((s.tail(20) * vv).mean())
        rows.append({
            "Symbol": row.Symbol,
            "last_price": float(s.iloc[-1]),
            "mom_12_1": _ret(s, min(252, len(s) - 22), skip=21),
            "mom_6m": _ret(s, 126),
            "mom_3m": _ret(s, 63),
            "rel_6m": rel6,
            "rel_3m": rel3,
            "dist_200d": float(s.iloc[-1] / ma200 - 1) if np.isfinite(ma200) and ma200 else np.nan,
            "ma50_vs_200": float(ma50 / ma200 - 1) if np.isfinite(ma50) and np.isfinite(ma200) and ma200 else np.nan,
            "from_52w_high": float(s.iloc[-1] / s.tail(252).max() - 1),
            "vol_63d": float(r.tail(63).std() * np.sqrt(252)) if len(r) >= 63 else np.nan,
            "rsi14": _rsi(s),
            "adv20": adv,
        })
    return pd.DataFrame(rows)


def sector_rotation(close: pd.DataFrame, universe: pd.DataFrame, tech: pd.DataFrame, sector_etfs: dict) -> pd.DataFrame:
    spy = close.get("SPY")
    rows = []
    for sector, etf in sector_etfs.items():
        etf_series = close.get(etf)
        rel6 = _ret(etf_series, 126) - _ret(spy, 126) if etf_series is not None and spy is not None else np.nan
        rel3 = _ret(etf_series, 63) - _ret(spy, 63) if etf_series is not None and spy is not None else np.nan
        symbols = universe.loc[universe["GICS Sector"].eq(sector), "Symbol"]
        subset = tech[tech["Symbol"].isin(symbols)]
        breadth = float((subset["dist_200d"] > 0).mean()) if not subset.empty else np.nan
        rows.append({"GICS Sector": sector, "sector_rel6": rel6, "sector_rel3": rel3, "sector_breadth": breadth})
    out = pd.DataFrame(rows)
    out["sector_rotation"] = weighted_available(out.assign(
        r6=percentile_score(out["sector_rel6"]),
        r3=percentile_score(out["sector_rel3"]),
        br=percentile_score(out["sector_breadth"]),
    ), [("r6", 0.45), ("r3", 0.25), ("br", 0.30)])
    return out


def fred_series(cache: Cache, name: str, series: str, hours: float) -> pd.DataFrame:
    cached = cache.load_csv(f"fred_{series}", hours)
    if cached is not None:
        df = cached
    else:
        response = requests.get(FRED_CSV.format(series=series), timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        cache.save_csv(f"fred_{series}", df)
    date_col = df.columns[0]
    value_col = df.columns[-1]
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), name: pd.to_numeric(df[value_col], errors="coerce")})
    return out.dropna().sort_values("date")


def _latest(df: pd.DataFrame, col: str) -> float:
    return float(df[col].dropna().iloc[-1]) if not df.empty and df[col].notna().any() else np.nan


def _percentile_last(df: pd.DataFrame, col: str, years: int = 10) -> float:
    if df.empty:
        return np.nan
    cutoff = df["date"].max() - pd.DateOffset(years=years)
    s = df.loc[df["date"] >= cutoff, col].dropna()
    if len(s) < 20:
        s = df[col].dropna()
    if s.empty:
        return np.nan
    return float(s.rank(pct=True).iloc[-1])


def macro_state(cache: Cache, hours: float) -> MacroState:
    data = {}
    rows = []
    for name, series in FRED_SERIES.items():
        try:
            df = fred_series(cache, name, series, hours)
            data[name] = df
            rows.append({"name": name, "series": series, "date": df["date"].iloc[-1], "value": _latest(df, name), "pct10y": _percentile_last(df, name)})
        except Exception:
            rows.append({"name": name, "series": series, "date": pd.NaT, "value": np.nan, "pct10y": np.nan})
    snap = pd.DataFrame(rows)
    vals = snap.set_index("name")["value"].to_dict()
    pcts = snap.set_index("name")["pct10y"].to_dict()

    cpi_df = data.get("cpi", pd.DataFrame())
    cpi_yoy = np.nan
    if not cpi_df.empty and len(cpi_df) >= 13:
        s = cpi_df.set_index("date")["cpi"].sort_index()
        cpi_yoy = float(s.iloc[-1] / s.iloc[-13] - 1) * 100
    snap.loc[snap["name"].eq("cpi"), "derived_yoy_pct"] = cpi_yoy

    sahm = vals.get("sahm", np.nan)
    curve = vals.get("curve_10y3m", np.nan)
    hy = vals.get("hy_oas", np.nan)
    nfci = vals.get("nfci", np.nan)
    real10 = vals.get("real10y", np.nan)
    cfnai = vals.get("cfnai", np.nan)

    recession = np.nanmean([
        min(max((sahm if np.isfinite(sahm) else 0) / 0.5, 0), 1),
        pcts.get("hy_oas", np.nan),
        pcts.get("nfci", np.nan),
        1.0 if np.isfinite(curve) and curve < 0 else 0.0,
        1 - pcts.get("cfnai", np.nan) if np.isfinite(pcts.get("cfnai", np.nan)) else np.nan,
    ])
    inflation = np.nanmean([
        min(max((cpi_yoy - 2.0) / 3.0, 0), 1) if np.isfinite(cpi_yoy) else np.nan,
        pcts.get("real10y", np.nan),
    ])
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
    return MacroState(regime, recession, inflation, liquidity, rate_pressure, snap)


def safe_growth(current, previous):
    if not np.isfinite(current) or not np.isfinite(previous) or abs(previous) < 1e-12:
        return np.nan
    return float(current / previous - 1)


def build_features(universe: pd.DataFrame, tech: pd.DataFrame, fund: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    df = universe.merge(tech, on="Symbol", how="left").merge(fund, on="Symbol", how="left").merge(sector, on="GICS Sector", how="left")
    df["market_cap"] = df["last_price"] * df["shares"]
    df["fcf_ttm"] = df["cfo_ttm"] - df["capex_ttm"]
    df["earnings_yield"] = df["net_income_ttm"] / df["market_cap"]
    df["fcf_yield"] = df["fcf_ttm"] / df["market_cap"]
    df["book_to_price"] = df["equity"] / df["market_cap"]
    df["op_margin"] = df["operating_income_ttm"] / df["revenue_ttm"]
    df["cfo_margin"] = df["cfo_ttm"] / df["revenue_ttm"]
    df["roa"] = df["net_income_ttm"] / df["assets"]
    df["roe"] = df["net_income_ttm"] / df["equity"]
    df["debt_to_assets"] = df["debt"] / df["assets"]
    df["cash_to_assets"] = df["cash"] / df["assets"]
    df["revenue_growth"] = [safe_growth(c, p) for c, p in zip(df["revenue_ttm"], df["revenue_prior_ttm"])]
    df["net_income_growth"] = [safe_growth(c, p) for c, p in zip(df["net_income_ttm"], df["net_income_prior_ttm"])]

    score_specs = {
        "mom_12_1": True, "mom_6m": True, "mom_3m": True, "rel_6m": True, "rel_3m": True,
        "dist_200d": True, "ma50_vs_200": True, "from_52w_high": True, "vol_63d": False,
        "earnings_yield": True, "fcf_yield": True, "book_to_price": True,
        "op_margin": True, "cfo_margin": True, "roa": True, "roe": True,
        "debt_to_assets": False, "cash_to_assets": True, "revenue_growth": True, "net_income_growth": True,
    }
    for col, hib in score_specs.items():
        df[f"{col}_score"] = blended_percentile(df, col, hib)

    df["momentum"] = weighted_available(df, [
        ("mom_12_1_score", .24), ("mom_6m_score", .20), ("mom_3m_score", .10),
        ("rel_6m_score", .18), ("rel_3m_score", .08), ("dist_200d_score", .08),
        ("ma50_vs_200_score", .06), ("from_52w_high_score", .06),
    ])
    df["low_vol"] = df["vol_63d_score"]
    financial = df["GICS Sector"].eq("Financials")
    reit = df["GICS Sector"].eq("Real Estate")
    regular = ~(financial | reit)
    df["value"] = np.nan
    df.loc[regular, "value"] = weighted_available(df.loc[regular], [("earnings_yield_score", .45), ("fcf_yield_score", .55)])
    df.loc[financial, "value"] = weighted_available(df.loc[financial], [("earnings_yield_score", .55), ("book_to_price_score", .45)])
    df.loc[reit, "value"] = weighted_available(df.loc[reit], [("earnings_yield_score", .65), ("book_to_price_score", .35)])
    df["quality"] = np.nan
    df.loc[regular, "quality"] = weighted_available(df.loc[regular], [
        ("op_margin_score", .20), ("cfo_margin_score", .20), ("roa_score", .20),
        ("debt_to_assets_score", .20), ("cash_to_assets_score", .20),
    ])
    df.loc[financial, "quality"] = weighted_available(df.loc[financial], [
        ("roe_score", .55), ("roa_score", .25), ("cash_to_assets_score", .20),
    ])
    df.loc[reit, "quality"] = weighted_available(df.loc[reit], [
        ("op_margin_score", .45), ("roe_score", .25), ("cash_to_assets_score", .30),
    ])
    df["growth"] = weighted_available(df, [("revenue_growth_score", .60), ("net_income_growth_score", .40)])
    df["extension_penalty"] = 0.0
    df.loc[df["rsi14"] > 80, "extension_penalty"] += ((df.loc[df["rsi14"] > 80, "rsi14"] - 80) / 20).clip(upper=1) * .08
    df.loc[df["dist_200d"] > .30, "extension_penalty"] += ((df.loc[df["dist_200d"] > .30, "dist_200d"] - .30) / .30).clip(upper=1) * .07
    factors = ["momentum", "quality", "value", "growth", "low_vol", "sector_rotation"]
    df["data_completeness"] = df[factors].notna().mean(axis=1)
    df["fundamental_completeness"] = df[["quality", "value", "growth"]].notna().mean(axis=1)
    return df


def score_stocks(df: pd.DataFrame, macro: MacroState, config: dict) -> pd.DataFrame:
    out = df.copy()
    weights = config["regime_weights"][macro.regime]
    tilts = config.get("macro_sector_tilts", {}).get(macro.regime, {})
    out["macro_sector_score"] = (out["GICS Sector"].map(tilts).fillna(0) + 1) / 2
    mapping = {
        "momentum": "momentum", "growth": "growth", "quality": "quality", "value": "value",
        "low_vol": "low_vol", "sector_rotation": "sector_rotation", "macro_sector": "macro_sector_score",
    }
    num = pd.Series(0.0, index=out.index)
    den = pd.Series(0.0, index=out.index)
    for key, col in mapping.items():
        w = float(weights.get(key, 0))
        mask = out[col].notna()
        num.loc[mask] += out.loc[mask, col] * w
        den.loc[mask] += w
    out["raw_score"] = num / den.replace(0, np.nan)
    out["score"] = ((out["raw_score"] - out["extension_penalty"]).clip(0, 1) * 100) * (.75 + .25 * out["data_completeness"])
    out["eligible"] = (out["data_completeness"] >= float(config.get("min_data_completeness", .45))) & (out["fundamental_completeness"] >= 1 / 3)
    if config.get("trend_gate", True):
        out["eligible"] &= ((out["dist_200d"] > 0) & (out["ma50_vs_200"] > -.02)).fillna(False)
    out["rank"] = out["score"].where(out["eligible"]).rank(ascending=False, method="min")
    return out.sort_values(["eligible", "score"], ascending=[False, False]).reset_index(drop=True)


def build_portfolio(ranked: pd.DataFrame, n: int, max_stock: float, max_sector: float) -> pd.DataFrame:
    picks = ranked[ranked["eligible"]].head(n).copy()
    if picks.empty:
        return picks
    raw = (1 / picks["vol_63d"].replace(0, np.nan)) * (picks["score"] / 100).clip(lower=.01)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    picks["weight"] = raw / raw.sum()
    for _ in range(40):
        before = picks["weight"].copy()
        over = picks["weight"] > max_stock
        if over.any():
            excess = (picks.loc[over, "weight"] - max_stock).sum()
            picks.loc[over, "weight"] = max_stock
            room = (~over) & (picks["weight"] < max_stock)
            if room.any() and picks.loc[room, "weight"].sum() > 0:
                picks.loc[room, "weight"] += excess * picks.loc[room, "weight"] / picks.loc[room, "weight"].sum()
        for sec, total in picks.groupby("GICS Sector")["weight"].sum().items():
            if total <= max_sector:
                continue
            idx = picks["GICS Sector"].eq(sec)
            excess = total - max_sector
            picks.loc[idx, "weight"] *= max_sector / total
            room = ~idx
            if room.any() and picks.loc[room, "weight"].sum() > 0:
                picks.loc[room, "weight"] += excess * picks.loc[room, "weight"] / picks.loc[room, "weight"].sum()
        if np.allclose(before, picks["weight"], atol=1e-10):
            break
    picks["weight"] /= picks["weight"].sum()
    return picks.sort_values("weight", ascending=False)


def print_summary(ranked: pd.DataFrame, portfolio: pd.DataFrame, macro: MacroState, top_n: int) -> None:
    print(f"\n=== MACRO REGIME ===\nRegime: {macro.regime}")
    print(f"Recession risk: {macro.recession_risk:.2f}" if np.isfinite(macro.recession_risk) else "Recession risk: N/A")
    cols = ["rank", "Symbol", "Security", "GICS Sector", "score", "momentum", "quality", "value", "growth", "low_vol", "sector_rotation", "data_completeness"]
    show = ranked[ranked["eligible"]].head(top_n)[cols].copy()
    print("\n=== TOP RANKED ===")
    print(show.round(3).to_string(index=False))
    if not portfolio.empty:
        p = portfolio[["Symbol", "Security", "GICS Sector", "score", "weight", "vol_63d"]].copy()
        p["weight"] *= 100
        print("\n=== REFERENCE PORTFOLIO (%) ===")
        print(p.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="S&P 500 multi-factor, sector-rotation and macro-regime screener")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--universe-csv", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--portfolio", type=int, default=None)
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    cache = Cache(Path(config.get("cache_dir", ".cache_sp500_screener")))
    hours = config.get("cache_hours", {})
    universe = get_universe(cache, float(hours.get("universe", 12)), args.universe_csv)
    symbols = universe["Yahoo Symbol"].tolist() + ["SPY"] + list(config["sector_etfs"].values())
    close, volume = download_prices(symbols, config.get("price_period", "2y"), int(config.get("price_batch_size", 80)))
    if close.empty:
        raise RuntimeError("Price download returned no data.")
    tech = price_features(close, volume, universe)
    sector = sector_rotation(close, universe, tech, config["sector_etfs"])
    fund = fetch_fundamentals(universe, cache, float(hours.get("sec", 24)))
    macro = macro_state(cache, float(hours.get("macro", 6)))
    ranked = score_stocks(build_features(universe, tech, fund, sector), macro, config)
    portfolio = build_portfolio(
        ranked,
        args.portfolio or int(config.get("portfolio_n", 10)),
        float(config.get("max_stock_weight", .12)),
        float(config.get("max_sector_weight", .30)),
    )
    ranked.to_csv(args.output, index=False)
    portfolio.to_csv("portfolio.csv", index=False)
    macro.snapshot.to_csv("macro_snapshot.csv", index=False)
    print_summary(ranked, portfolio, macro, args.top or int(config.get("top_n", 20)))
    print(f"\nSaved: {args.output}, portfolio.csv, macro_snapshot.csv")


if __name__ == "__main__":
    main()
