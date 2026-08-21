from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from historical_universe_compare import PIERRE_RAW, load_hans, norm_symbol, parse_pierre

OUT = Path("data/research")
UA = {"User-Agent": "regime-factor-investing research"}
MAX_POST_SNAPSHOT_LAG_DAYS = 45

# Curated, high-authority spot checks from S&P Global press releases. This is
# deliberately an audit sample rather than a claim that these events form a
# complete constituent-change history.
CHECKS = [
    {
        "effective_date": "2018-05-31",
        "added": "ABMD",
        "deleted": "WYN",
        "source": "https://press.spglobal.com/2018-05-25-ABIOMED-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2018-12-03",
        "added": "LW",
        "deleted": "COL",
        "source": "https://press.spglobal.com/2018-11-26-Lamb-Weston-Holdings-Maxim-Integrated-Products-and-Diamondback-Energy-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2018-12-03",
        "added": "MXIM",
        "deleted": "AET",
        "source": "https://press.spglobal.com/2018-11-26-Lamb-Weston-Holdings-Maxim-Integrated-Products-and-Diamondback-Energy-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2020-10-07",
        "added": "POOL",
        "deleted": "ETFC",
        "source": "https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20201001-1231161/1231161_dlph400etfc500.pdf",
    },
    {
        "effective_date": "2022-02-03",
        "added": "CEG",
        "deleted": "GPS",
        "source": "https://press.spglobal.com/2022-01-26-Constellation-Energy-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2023-03-15",
        "added": "PODD",
        "deleted": "SIVB",
        "source": "https://press.spglobal.com/2023-03-10-Insulet-Set-to-Join-S-P-500",
    },
    {
        "effective_date": "2023-03-15",
        "added": "BG",
        "deleted": "SBNY",
        "source": "https://press.spglobal.com/2023-03-13-Bunge-Set-to-Join-S-P-500",
    },
    {
        "effective_date": "2023-10-18",
        "added": "LULU",
        "deleted": "ATVI",
        "source": "https://press.spglobal.com/2023-10-13-Lululemon-Athletica-Hubbell-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2023-10-18",
        "added": "HUBB",
        "deleted": "OGN",
        "source": "https://press.spglobal.com/2023-10-13-Lululemon-Athletica-Hubbell-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
]


def hans_members_before(hans: pd.DataFrame, effective: pd.Timestamp) -> tuple[pd.Timestamp, set[str]]:
    eligible = hans[hans["date"] < effective]
    if eligible.empty:
        return pd.NaT, set()
    row = eligible.iloc[-1]
    return pd.Timestamp(row["date"]), set(row["members"])


def hans_members_after(
    hans: pd.DataFrame,
    effective: pd.Timestamp,
    max_lag_days: int = MAX_POST_SNAPSHOT_LAG_DAYS,
) -> tuple[pd.Timestamp, set[str]]:
    upper = effective + pd.Timedelta(days=max_lag_days)
    eligible = hans[(hans["date"] >= effective) & (hans["date"] <= upper)]
    if eligible.empty:
        return pd.NaT, set()
    row = eligible.iloc[0]
    return pd.Timestamp(row["date"]), set(row["members"])


def pierre_month(period: pd.Period) -> set[str]:
    url = PIERRE_RAW.format(yyyymm=period.strftime("%Y%m"))
    r = requests.get(url, timeout=45, headers=UA)
    r.raise_for_status()
    return parse_pierre(r.content)


def verify_source_url(url: str) -> tuple[bool, int | None]:
    try:
        r = requests.get(url, timeout=45, headers=UA, allow_redirects=True)
        return bool(r.status_code == 200 and len(r.content) > 100), int(r.status_code)
    except Exception:
        return False, None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hans = load_hans()
    pierre_cache: dict[str, set[str]] = {}
    rows = []

    for check in CHECKS:
        effective = pd.Timestamp(check["effective_date"])
        pre_date, pre = hans_members_before(hans, effective)
        post_date, post = hans_members_after(hans, effective)
        post_available = pd.notna(post_date)
        post_lag = int((post_date - effective).days) if post_available else None

        period = effective.to_period("M")
        key = str(period)
        if key not in pierre_cache:
            try:
                pierre_cache[key] = pierre_month(period)
            except Exception:
                pierre_cache[key] = set()
        pset = pierre_cache[key]

        add = norm_symbol(check["added"])
        delete = norm_symbol(check["deleted"])
        source_ok, source_status = verify_source_url(check["source"])
        pre_available = pd.notna(pre_date)
        hans_pass = bool(
            pre_available
            and post_available
            and add not in pre
            and add in post
            and delete in pre
            and delete not in post
        )
        pierre_pass = bool(pset) and add in pset and delete not in pset
        rows.append({
            **check,
            "source_http_ok": source_ok,
            "source_http_status": source_status,
            "hans_pre_snapshot": pre_date.date().isoformat() if pre_available else "",
            "hans_post_snapshot": post_date.date().isoformat() if post_available else "",
            "hans_post_snapshot_lag_days": post_lag,
            "hans_pre_snapshot_available": pre_available,
            "hans_post_snapshot_available": post_available,
            "hans_added_absent_pre": add not in pre if pre_available else False,
            "hans_added_present_post": add in post if post_available else False,
            "hans_deleted_present_pre": delete in pre if pre_available else False,
            "hans_deleted_absent_post": delete not in post if post_available else False,
            "hans_pass": hans_pass,
            "pierre_month": key,
            "pierre_added_present": add in pset,
            "pierre_deleted_absent": delete not in pset,
            "pierre_pass": pierre_pass,
            "both_sources_pass": bool(hans_pass and pierre_pass),
        })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "official_sp500_change_audit.csv", index=False)
    auditable = detail[detail["hans_pre_snapshot_available"] & detail["hans_post_snapshot_available"]]
    summary = pd.DataFrame([{
        "checks": len(detail),
        "official_urls_reachable": int(detail["source_http_ok"].sum()),
        "hans_auditable_checks": len(auditable),
        "hans_pass_rate": float(auditable["hans_pass"].mean()) if len(auditable) else float("nan"),
        "pierre_pass_rate": float(detail["pierre_pass"].mean()),
        "both_pass_rate": float(auditable["both_sources_pass"].mean()) if len(auditable) else float("nan"),
        "max_hans_post_snapshot_lag_days": int(auditable["hans_post_snapshot_lag_days"].max()) if len(auditable) else None,
        "all_auditable_checks_pass": bool(auditable["both_sources_pass"].all()) if len(auditable) else False,
        "scope_note": "Curated official S&P Global spot checks only; not a complete membership-history certification. Hans is scored only when both a pre-effective and first post-effective snapshot are available within 45 days.",
    }])
    summary.to_csv(OUT / "official_sp500_change_audit_summary.csv", index=False)
    print(detail.to_string(index=False), flush=True)
    print("\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
