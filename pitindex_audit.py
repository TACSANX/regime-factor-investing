from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

import pitindex
from historical_universe_compare import PIERRE_RAW, norm_symbol, parse_pierre
from official_sp500_change_audit import CHECKS

OUT = Path("data/research")
UA = {"User-Agent": "regime-factor-investing research"}
START = pd.Period("2018-01", freq="M")
END = pd.Period("2026-04", freq="M")


def pit_members(date: pd.Timestamp) -> set[str]:
    df = pitindex.get_constituents(date.date().isoformat(), index="sp500")
    return {norm_symbol(x) for x in df["ticker"].dropna().astype(str)}


def pierre_members(period: pd.Period) -> set[str]:
    url = PIERRE_RAW.format(yyyymm=period.strftime("%Y%m"))
    r = requests.get(url, timeout=45, headers=UA)
    r.raise_for_status()
    return {norm_symbol(x) for x in parse_pierre(r.content)}


def event_audit() -> pd.DataFrame:
    rows = []
    for check in CHECKS:
        effective = pd.Timestamp(check["effective_date"])
        pre = pit_members(effective - pd.Timedelta(days=1))
        post = pit_members(effective)
        add = norm_symbol(check["added"])
        delete = norm_symbol(check["deleted"])
        rows.append({
            "effective_date": check["effective_date"],
            "added": add,
            "deleted": delete,
            "added_absent_pre": add not in pre,
            "added_present_post": add in post,
            "deleted_present_pre": delete in pre,
            "deleted_absent_post": delete not in post,
            "pass": bool(add not in pre and add in post and delete in pre and delete not in post),
            "source": check["source"],
        })
    return pd.DataFrame(rows)


def monthly_compare() -> pd.DataFrame:
    rows = []
    for period in pd.period_range(START, END, freq="M"):
        month_end = period.end_time.normalize()
        pit = pit_members(month_end)
        try:
            pierre = pierre_members(period)
        except Exception:
            pierre = set()
        union = pit | pierre
        inter = pit & pierre
        rows.append({
            "month": str(period),
            "pitindex_members": len(pit),
            "pierre_members": len(pierre),
            "intersection": len(inter),
            "union": len(union),
            "jaccard": len(inter) / len(union) if union else float("nan"),
            "pit_only": ";".join(sorted(pit - pierre)),
            "pierre_only": ";".join(sorted(pierre - pit)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = event_audit()
    monthly = monthly_compare()
    events.to_csv(OUT / "pitindex_official_event_audit.csv", index=False)
    monthly.to_csv(OUT / "pitindex_pierre_monthly_compare.csv", index=False)

    info = pitindex.info(index="sp500")
    summary = pd.DataFrame([{
        "official_event_checks": len(events),
        "official_event_passes": int(events["pass"].sum()),
        "official_event_pass_rate": float(events["pass"].mean()),
        "months_compared": len(monthly),
        "mean_jaccard_vs_pierre": float(monthly["jaccard"].mean()),
        "median_jaccard_vs_pierre": float(monthly["jaccard"].median()),
        "min_jaccard_vs_pierre": float(monthly["jaccard"].min()),
        "months_ge_99pct": int((monthly["jaccard"] >= 0.99).sum()),
        "pitindex_data_age_days": info.get("data_age_days"),
        "pitindex_is_stale": info.get("is_stale"),
        "pitindex_reconciliation_diff_ratio": info.get("reconciliation_diff_ratio"),
        "candidate_for_primary_pit_universe": bool(
            events["pass"].all()
            and (monthly["jaccard"] >= 0.98).mean() >= 0.95
        ),
        "note": "Research audit only. Passing does not validate delisted price coverage or historical GICS metadata.",
    }])
    summary.to_csv(OUT / "pitindex_audit_summary.csv", index=False)
    print(events.to_string(index=False), flush=True)
    print("\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
