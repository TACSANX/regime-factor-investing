from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


def _capped_proportional(raw: np.ndarray, caps: np.ndarray, total: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    caps = np.asarray(caps, dtype=float)
    if len(raw) != len(caps):
        raise ValueError("raw/caps length mismatch")
    if total < -1e-12 or caps.sum() + 1e-10 < total:
        raise ValueError("Infeasible caps for requested total")
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 1.0)
    w = np.zeros(len(raw), dtype=float)
    active = np.ones(len(raw), dtype=bool)
    remaining = float(total)

    for _ in range(len(raw) + 5):
        if remaining <= 1e-12:
            break
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        capacity = caps[idx] - w[idx]
        usable = capacity > 1e-12
        idx = idx[usable]
        if len(idx) == 0:
            break
        denom = raw[idx].sum()
        proposed = remaining * (raw[idx] / denom if denom > 0 else np.full(len(idx), 1 / len(idx)))
        room = caps[idx] - w[idx]
        over = proposed > room + 1e-12
        if not over.any():
            w[idx] += proposed
            remaining = 0.0
            break
        fixed = idx[over]
        add = caps[fixed] - w[fixed]
        w[fixed] += add
        remaining -= float(add.sum())
        active[fixed] = False

    residual = total - float(w.sum())
    if residual > 1e-9:
        room = caps - w
        for i in np.argsort(-room):
            if residual <= 1e-12:
                break
            add = min(residual, max(0.0, float(room[i])))
            w[i] += add
            residual -= add
    if abs(float(w.sum()) - total) > 1e-7:
        raise ValueError("Unable to allocate weights within caps")
    return w


def build_portfolio_strict(ranked: pd.DataFrame, n: int, max_stock: float, max_sector: float) -> pd.DataFrame:
    if n <= 0:
        return ranked.iloc[0:0].copy()
    if n * max_stock < 1 - 1e-9:
        raise ValueError(f"Infeasible stock cap: n={n}, max_stock={max_stock}")
    if max_sector <= 0 or max_stock <= 0:
        raise ValueError("Caps must be positive")

    eligible = ranked[ranked["eligible"]].copy()
    if eligible.empty:
        return eligible

    # Avoid a top-N set that cannot satisfy the sector cap.  With a 30% sector
    # cap and 12% stock cap, three names per sector are sufficient capacity.
    max_names_per_sector = max(1, math.ceil(max_sector / max_stock))
    selected_idx: list[int] = []
    sector_counts: dict[str, int] = defaultdict(int)
    for idx, row in eligible.iterrows():
        sec = str(row["GICS Sector"])
        if sector_counts[sec] >= max_names_per_sector:
            continue
        selected_idx.append(idx)
        sector_counts[sec] += 1
        if len(selected_idx) >= n:
            break

    if len(selected_idx) < n:
        # If the universe is unusually narrow, allow extra names only when the
        # final sector-capacity check remains feasible.
        for idx, row in eligible.iterrows():
            if idx in selected_idx:
                continue
            selected_idx.append(idx)
            if len(selected_idx) >= n:
                break

    picks = eligible.loc[selected_idx].head(n).copy()
    if picks.empty:
        return picks

    raw = (1 / picks["vol_63d"].replace(0, np.nan)) * (picks["score"] / 100).clip(lower=0.01)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(float)

    sectors = picks["GICS Sector"].astype(str)
    sector_names = list(dict.fromkeys(sectors.tolist()))
    sector_raw = np.array([raw[sectors.eq(sec)].sum() for sec in sector_names], dtype=float)
    sector_caps = np.array([
        min(max_sector, int(sectors.eq(sec).sum()) * max_stock)
        for sec in sector_names
    ], dtype=float)
    if sector_caps.sum() < 1 - 1e-9:
        raise ValueError(
            "Selected names cannot satisfy sector/stock caps; increase n or diversify candidate selection"
        )

    sector_budget = _capped_proportional(sector_raw, sector_caps, 1.0)
    weights = pd.Series(0.0, index=picks.index, dtype=float)
    for sec, budget in zip(sector_names, sector_budget):
        idx = picks.index[sectors.eq(sec)]
        local_raw = raw.loc[idx].to_numpy(dtype=float)
        local_caps = np.full(len(idx), max_stock, dtype=float)
        local_w = _capped_proportional(local_raw, local_caps, float(budget))
        weights.loc[idx] = local_w

    picks["weight"] = weights
    if abs(float(picks["weight"].sum()) - 1.0) > 1e-7:
        raise AssertionError("Portfolio weights do not sum to one")
    if float(picks["weight"].max()) > max_stock + 1e-7:
        raise AssertionError("Stock cap violated")
    if float(picks.groupby("GICS Sector")["weight"].sum().max()) > max_sector + 1e-7:
        raise AssertionError("Sector cap violated")
    return picks.sort_values("weight", ascending=False)
