from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from historical_universe_compare import PIERRE_RAW, load_hans, norm_symbol, parse_pierre

OUT = Path("data/research")
UA = {"User-Agent": "regime-factor-investing research"}
MAX_POST_SNAPSHOT_LAG_DAYS = 45

# Curated, high-authority spot checks from S&P Global press releases. The
# sample deliberately spans one-off replacements, merger deletions and several
# quarterly rebalances. It is an audit sample, not a complete official history.
CHECKS = [
    # 2018
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
    # 2019
    {
        "effective_date": "2019-09-23",
        "added": "CDW",
        "deleted": "TSS",
        "source": "https://press.spglobal.com/2019-09-17-CDW-Set-to-Join-S-P-500",
    },
    {
        "effective_date": "2019-09-26",
        "added": "NVR",
        "deleted": "JEF",
        "source": "https://press.spglobal.com/2019-09-20-NVR-Set-to-Join-S-P-500-Jefferies-Financial-Group-II-VI-to-Join-S-P-MidCap-400-Callon-Petroleum-PriceSmart-to-Join-S-P-SmallCap-600",
    },
    {
        "effective_date": "2019-10-03",
        "added": "LVS",
        "deleted": "NKTR",
        "source": "https://press.spglobal.com/2019-09-26-Las-Vegas-Sands-Set-to-Join-S-P-500-Nektar-Therapeutics-to-Join-S-P-MidCap-400-The-Pennant-Group-to-Join-S-P-SmallCap-600",
    },
    # 2020 quarterly rebalance plus a later one-off replacement.
    {
        "effective_date": "2020-09-21",
        "added": "ETSY",
        "deleted": "HRB",
        "source": "https://press.spglobal.com/2020-09-04-Etsy-Teradyne-and-Catalent-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2020-09-21",
        "added": "TER",
        "deleted": "COTY",
        "source": "https://press.spglobal.com/2020-09-04-Etsy-Teradyne-and-Catalent-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2020-09-21",
        "added": "CTLT",
        "deleted": "KSS",
        "source": "https://press.spglobal.com/2020-09-04-Etsy-Teradyne-and-Catalent-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2020-10-07",
        "added": "POOL",
        "deleted": "ETFC",
        "source": "https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20201001-1231161/1231161_dlph400etfc500.pdf",
    },
    # 2021 quarterly rebalance.
    {
        "effective_date": "2021-03-22",
        "added": "NXPI",
        "deleted": "FLS",
        "source": "https://press.spglobal.com/2021-03-12-NXP-Semiconductors-Penn-National-Gaming-Generac-Holdings-and-Caesars-Entertainment-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-S-P-SmallCap-600-and-S-P-100",
    },
    {
        "effective_date": "2021-03-22",
        "added": "PENN",
        "deleted": "SLG",
        "source": "https://press.spglobal.com/2021-03-12-NXP-Semiconductors-Penn-National-Gaming-Generac-Holdings-and-Caesars-Entertainment-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-S-P-SmallCap-600-and-S-P-100",
    },
    {
        "effective_date": "2021-03-22",
        "added": "GNRC",
        "deleted": "XRX",
        "source": "https://press.spglobal.com/2021-03-12-NXP-Semiconductors-Penn-National-Gaming-Generac-Holdings-and-Caesars-Entertainment-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-S-P-SmallCap-600-and-S-P-100",
    },
    {
        "effective_date": "2021-03-22",
        "added": "CZR",
        "deleted": "VNT",
        "source": "https://press.spglobal.com/2021-03-12-NXP-Semiconductors-Penn-National-Gaming-Generac-Holdings-and-Caesars-Entertainment-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-S-P-SmallCap-600-and-S-P-100",
    },
    # 2022-2023 special replacements.
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
    # 2024: four quarterly rebalances, selected S&P 500 pairs.
    {
        "effective_date": "2024-03-18",
        "added": "SMCI",
        "deleted": "WHR",
        "source": "https://press.spglobal.com/2024-03-01-Super-Micro-Computer-and-Deckers-Outdoor-Set-to-Join-S-P-500-Others-to-Join-S-P-100%2C-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-03-18",
        "added": "DECK",
        "deleted": "ZION",
        "source": "https://press.spglobal.com/2024-03-01-Super-Micro-Computer-and-Deckers-Outdoor-Set-to-Join-S-P-500-Others-to-Join-S-P-100%2C-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-06-24",
        "added": "KKR",
        "deleted": "RHI",
        "source": "https://press.spglobal.com/2024-06-07-KKR%2C-CrowdStrike-Holdings-and-GoDaddy-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-06-24",
        "added": "CRWD",
        "deleted": "CMA",
        "source": "https://press.spglobal.com/2024-06-07-KKR%2C-CrowdStrike-Holdings-and-GoDaddy-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-06-24",
        "added": "GDDY",
        "deleted": "ILMN",
        "source": "https://press.spglobal.com/2024-06-07-KKR%2C-CrowdStrike-Holdings-and-GoDaddy-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-09-23",
        "added": "PLTR",
        "deleted": "AAL",
        "source": "https://press.spglobal.com/2024-09-06-Palantir-Technologies%2C-Dell-Technologies%2C-and-Erie-Indemnity-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-09-23",
        "added": "DELL",
        "deleted": "ETSY",
        "source": "https://press.spglobal.com/2024-09-06-Palantir-Technologies%2C-Dell-Technologies%2C-and-Erie-Indemnity-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-09-23",
        "added": "ERIE",
        "deleted": "BIO",
        "source": "https://press.spglobal.com/2024-09-06-Palantir-Technologies%2C-Dell-Technologies%2C-and-Erie-Indemnity-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-12-23",
        "added": "APO",
        "deleted": "QRVO",
        "source": "https://press.spglobal.com/2024-12-06-Apollo-Global-Management-and-Workday-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    {
        "effective_date": "2024-12-23",
        "added": "WDAY",
        "deleted": "AMTM",
        "source": "https://press.spglobal.com/2024-12-06-Apollo-Global-Management-and-Workday-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600",
    },
    # 2025 one-off merger replacement.
    {
        "effective_date": "2025-05-19",
        "added": "COIN",
        "deleted": "DFS",
        "source": "https://press.spglobal.com/2025-05-12-Coinbase-Global-Set-to-Join-S-P-500",
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
