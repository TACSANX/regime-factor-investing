from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax"
DATES = ["20180131", "20181231", "20201231", "20221230", "20241231", "20251231"]
OUT = Path("data/research")


def inspect_csv(content: bytes) -> dict:
    # iShares CSVs have metadata lines before the holdings table and may include
    # a UTF-8 BOM.  Do not assume a fixed number of metadata rows.
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
    for date in DATES:
        params = {
            "fileType": "csv",
            "fileName": "IVV_holdings",
            "dataType": "fund",
            "asOfDate": date,
        }
        try:
            r = requests.get(URL, params=params, headers=headers, timeout=30)
            rec = {"requested_as_of": date, "status_code": r.status_code, "final_url": r.url}
            rec.update(inspect_csv(r.content))
        except Exception as exc:
            rec = {"requested_as_of": date, "status_code": -1, "error": repr(exc)}
        rows.append(rec)
        print(rec, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ivv_history_probe.csv", index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    main()
