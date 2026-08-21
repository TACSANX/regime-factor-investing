from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd
import requests

URLS = {
    "legacy_ajax": "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax",
    "latest_csv": "https://www.ishares.com/us/products/239726/ishares-core-s-p-500-etf/latest-holdings.csv",
}
DATES = ["20180131", "20181231", "20201231", "20221230", "20241231", "20251231"]
OUT = Path("data/research")


def inspect_csv(content: bytes) -> dict:
    text = content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        normalized = line.replace('"', "")
        if normalized.startswith("Ticker,Name,") or ("Ticker" in normalized and "Sector" in normalized and "Weight" in normalized):
            header_idx = i
            break
    holdings_date = None
    m = re.search(r"Fund Holdings as of,?\s*\"?([^\r\n\"]+)", text, flags=re.I)
    if m:
        holdings_date = m.group(1).strip().strip(",")
    rows = 0
    equity_rows = 0
    if header_idx is not None:
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
            rows = len(df)
            if "Asset Class" in df.columns:
                equity_rows = int(df["Asset Class"].astype(str).str.contains("Equity", case=False, na=False).sum())
            else:
                equity_rows = rows
        except Exception:
            pass
    return {
        "content_bytes": len(content),
        "looks_like_html": text.lstrip().lower().startswith("<!doctype html") or text.lstrip().lower().startswith("<html"),
        "header_found": header_idx is not None,
        "holdings_date_text": holdings_date,
        "parsed_rows": rows,
        "parsed_equity_rows": equity_rows,
        "first_120_chars": text[:120].replace("\n", " ").replace("\r", " "),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    headers = {"User-Agent": "regime-factor-investing research probe"}
    for endpoint, url in URLS.items():
        for date in DATES:
            if endpoint == "legacy_ajax":
                params = {"fileType": "csv", "fileName": "IVV_holdings", "dataType": "fund", "asOfDate": date}
            else:
                params = {"asOfDate": date}
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)
                rec = {"endpoint": endpoint, "requested_as_of": date, "status_code": r.status_code, "final_url": r.url}
                rec.update(inspect_csv(r.content))
            except Exception as exc:
                rec = {"endpoint": endpoint, "requested_as_of": date, "status_code": -1, "error": repr(exc)}
            rows.append(rec)
            print(rec, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ivv_history_probe.csv", index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    main()
