from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("data/research")
HANS_RAW = "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp_500_historical_components.csv"
PIERRE_RAW = "https://raw.githubusercontent.com/pierrebrunelle/sp500-historical-constituents/main/data/spy{yyyymm}"
START = pd.Period("2018-01", freq="M")
END = pd.Period("2026-04", freq="M")


def norm_symbol(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def get(url: str) -> bytes:
    r = requests.get(url, timeout=60, headers={"User-Agent": "regime-factor-investing research"})
    r.raise_for_status()
    return r.content


def load_hans() -> pd.DataFrame:
    raw = get(HANS_RAW)
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    required = {"date", "tickers"}
    if not required.issubset({str(c).lower() for c in df.columns}):
        raise RuntimeError(f"Unexpected hanshof columns: {list(df.columns)}")
    colmap = {str(c).lower(): c for c in df.columns}
    df = df.rename(columns={colmap["date"]: "date", colmap["tickers"]: "tickers"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    df["members"] = df["tickers"].fillna("").map(
        lambda s: {norm_symbol(x) for x in str(s).split(",") if str(x).strip()}
    )
    return df[["date", "members"]]


def hans_members(df: pd.DataFrame, period: pd.Period) -> tuple[pd.Timestamp, set[str]]:
    month_end = period.end_time.normalize()
    eligible = df[df["date"] <= month_end]
    if eligible.empty:
        return pd.NaT, set()
    row = eligible.iloc[-1]
    return pd.Timestamp(row["date"]), set(row["members"])


def parse_pierre(content: bytes) -> set[str]:
    text = content.decode("utf-8-sig", errors="replace")
    members: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Files are tab-delimited: ticker, company name, equal-weight placeholder.
        # Keep a comma fallback in case the upstream format changes.
        token = line.split("\t", 1)[0].split(",", 1)[0].strip().strip('"')
        if token and token.lower() not in {"ticker", "symbol"}:
            members.add(norm_symbol(token))
    return members


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hans = load_hans()
    summary_rows = []
    mismatch_rows = []

    for period in pd.period_range(START, END, freq="M"):
        yyyymm = period.strftime("%Y%m")
        hdate, hset = hans_members(hans, period)
        try:
            pset = parse_pierre(get(PIERRE_RAW.format(yyyymm=yyyymm)))
            pierre_ok = True
        except Exception as exc:
            print(f"Pierre missing {yyyymm}: {exc}", flush=True)
            pset = set()
            pierre_ok = False

        inter = hset & pset
        union = hset | pset
        only_h = sorted(hset - pset)
        only_p = sorted(pset - hset)
        jaccard = len(inter) / len(union) if union else np.nan
        summary_rows.append({
            "month": str(period),
            "hans_snapshot_date": hdate.date().isoformat() if pd.notna(hdate) else "",
            "hans_count": len(hset),
            "pierre_count": len(pset),
            "pierre_available": pierre_ok,
            "intersection_count": len(inter),
            "union_count": len(union),
            "hans_only_count": len(only_h),
            "pierre_only_count": len(only_p),
            "jaccard": jaccard,
            "agreement_min_denominator": len(inter) / min(len(hset), len(pset)) if hset and pset else np.nan,
        })
        for sym in only_h:
            mismatch_rows.append({"month": str(period), "symbol": sym, "side": "hans_only"})
        for sym in only_p:
            mismatch_rows.append({"month": str(period), "symbol": sym, "side": "pierre_only"})
        print(
            f"{period}: hans={len(hset)} pierre={len(pset)} inter={len(inter)} "
            + (f"jaccard={jaccard:.4f}" if np.isfinite(jaccard) else "jaccard=NA"),
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    mismatches = pd.DataFrame(mismatch_rows, columns=["month", "symbol", "side"])
    summary.to_csv(OUT / "historical_universe_comparison.csv", index=False)
    mismatches.to_csv(OUT / "historical_universe_mismatches.csv", index=False)

    valid = summary[summary["pierre_available"] & summary["jaccard"].notna()].copy()
    median_jaccard = float(valid["jaccard"].median()) if len(valid) else np.nan
    quality = pd.DataFrame([{
        "months_compared": len(valid),
        "mean_jaccard": valid["jaccard"].mean(),
        "median_jaccard": median_jaccard,
        "min_jaccard": valid["jaccard"].min(),
        "months_jaccard_ge_0_99": int((valid["jaccard"] >= 0.99).sum()),
        "months_jaccard_ge_0_98": int((valid["jaccard"] >= 0.98).sum()),
        "max_hans_only": int(valid["hans_only_count"].max()) if len(valid) else np.nan,
        "max_pierre_only": int(valid["pierre_only_count"].max()) if len(valid) else np.nan,
        "usable_for_survivorship_free_test": bool(len(valid) >= 90 and np.isfinite(median_jaccard) and median_jaccard >= 0.98),
        "source_note_hans": "MIT; pre-2019 history seeded from fja05680/sp500, then daily Wikipedia snapshots",
        "source_note_pierre": "MIT; reconstructs monthly sets backward from Wikipedia current constituents and change table",
        "independence_note": "Not fully independent: both ultimately rely on Wikipedia-related constituent history; comparison detects implementation/data disagreements, not common-source errors.",
    }])
    quality.to_csv(OUT / "historical_universe_quality.csv", index=False)


if __name__ == "__main__":
    main()
