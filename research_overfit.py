from __future__ import annotations

import itertools
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
OUT = ROOT
BLOCKS = 8


def sharpe_periodic(x: pd.Series) -> float:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 3:
        return np.nan
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd) if sd > 0 else np.nan


def expected_max_sharpe_threshold(sharpes: pd.Series) -> float:
    """Bailey/Lopez de Prado expected maximum under N zero-skill trials.

    The input is a cross-section of *per-period* Sharpe estimates.  The raw
    number of tested variants is deliberately used as N.  Because strategies
    are correlated this is a diagnostic rather than an exact independent-trial
    calculation; the output file reports that limitation explicitly.
    """
    s = pd.Series(sharpes, dtype=float).dropna()
    n = len(s)
    if n <= 1:
        return 0.0
    sigma = float(s.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    nd = NormalDist()
    gamma = 0.5772156649015329
    e = math.e
    z = (1.0 - gamma) * nd.inv_cdf(1.0 - 1.0 / n) + gamma * nd.inv_cdf(1.0 - 1.0 / (n * e))
    return sigma * z


def dsr_probability(active: pd.Series, sr0: float) -> float:
    x = pd.Series(active, dtype=float).dropna()
    n = len(x)
    if n < 3:
        return np.nan
    sr = sharpe_periodic(x)
    if not np.isfinite(sr):
        return np.nan
    skew = float(x.skew())
    raw_kurtosis = float(x.kurt()) + 3.0  # pandas reports excess kurtosis
    variance_term = 1.0 - skew * sr + ((raw_kurtosis - 1.0) / 4.0) * (sr**2)
    if not np.isfinite(variance_term) or variance_term <= 0:
        return np.nan
    z = (sr - sr0) * math.sqrt(n - 1.0) / math.sqrt(variance_term)
    return float(NormalDist().cdf(z))


def contiguous_blocks(n: int, blocks: int) -> list[np.ndarray]:
    if blocks % 2:
        raise ValueError("CSCV requires an even number of blocks")
    if n < blocks * 2:
        raise ValueError("Too few observations for requested CSCV blocks")
    return [a for a in np.array_split(np.arange(n), blocks) if len(a)]


def cscv_pbo(active_matrix: pd.DataFrame, blocks: int = BLOCKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combinatorially symmetric cross-validation PBO diagnostic.

    For each half/half block split, choose the strategy with the highest
    in-sample active Sharpe.  Rank that same strategy by out-of-sample active
    Sharpe. PBO is the fraction whose OOS relative rank is at/below 0.5.
    """
    matrix = active_matrix.dropna(axis=0, how="any").copy()
    names = list(matrix.columns)
    arr = matrix.to_numpy(dtype=float)
    b = contiguous_blocks(len(matrix), blocks)
    half = blocks // 2
    rows: list[dict] = []

    # Complementary splits are intentionally all retained; this matches the
    # symmetric CSCV construction and makes the diagnostic transparent.
    for split_id, combo in enumerate(itertools.combinations(range(blocks), half), start=1):
        is_idx = np.concatenate([b[i] for i in combo])
        oos_idx = np.concatenate([b[i] for i in range(blocks) if i not in combo])
        is_scores = np.array([sharpe_periodic(pd.Series(arr[is_idx, j])) for j in range(arr.shape[1])])
        if not np.isfinite(is_scores).any():
            continue
        winner_j = int(np.nanargmax(is_scores))
        oos_scores = np.array([sharpe_periodic(pd.Series(arr[oos_idx, j])) for j in range(arr.shape[1])])
        finite = np.isfinite(oos_scores)
        if not finite[winner_j] or finite.sum() < 2:
            continue

        # 1 = worst, m = best; mid-rank scaling avoids exactly 0/1 logits.
        finite_scores = pd.Series(oos_scores[finite])
        finite_ranks = finite_scores.rank(method="average", ascending=True)
        winner_pos = int(np.flatnonzero(finite)[np.where(np.flatnonzero(finite) == winner_j)[0][0]])
        winner_rank = float(finite_ranks.iloc[list(np.flatnonzero(finite)).index(winner_pos)])
        m = int(finite.sum())
        omega = (winner_rank - 0.5) / m
        omega = float(np.clip(omega, 1e-12, 1 - 1e-12))
        logit = float(math.log(omega / (1.0 - omega)))
        rows.append({
            "split": split_id,
            "is_blocks": ";".join(map(str, combo)),
            "is_winner": names[winner_j],
            "is_winner_sharpe_monthly": float(is_scores[winner_j]),
            "oos_winner_sharpe_monthly": float(oos_scores[winner_j]),
            "oos_relative_rank": omega,
            "logit": logit,
            "overfit_event": bool(omega <= 0.5),
        })

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "candidate_count": len(names),
        "common_months": len(matrix),
        "blocks": blocks,
        "splits": len(detail),
        "pbo": float(detail["overfit_event"].mean()) if len(detail) else np.nan,
        "median_oos_relative_rank": float(detail["oos_relative_rank"].median()) if len(detail) else np.nan,
        "median_logit": float(detail["logit"].median()) if len(detail) else np.nan,
        "interpretation": "PBO is the fraction of CSCV splits where the in-sample winner ranks at/below the OOS median; lower is better.",
    }])
    return detail, summary


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])
    specs = pd.read_csv(ROOT / "strategy_specs.csv")
    candidates = specs["strategy"].dropna().astype(str).tolist()

    m = monthly[monthly["strategy"].isin(candidates)][["signal_date", "strategy", "net_return"]]
    wide = m.pivot(index="signal_date", columns="strategy", values="net_return").sort_index()
    spy = bench.set_index("signal_date")["spy_return"].reindex(wide.index)
    active = wide.sub(spy, axis=0)

    periodic_sharpes = active.apply(sharpe_periodic)
    sr0 = expected_max_sharpe_threshold(periodic_sharpes)
    rows = []
    for name in active.columns:
        x = active[name].dropna()
        sr = sharpe_periodic(x)
        rows.append({
            "strategy": name,
            "months": len(x),
            "active_sharpe_monthly": sr,
            "active_sharpe_annualized": sr * math.sqrt(12) if np.isfinite(sr) else np.nan,
            "selection_threshold_monthly": sr0,
            "selection_threshold_annualized": sr0 * math.sqrt(12),
            "dsr_probability": dsr_probability(x, sr0),
            "passes_dsr_95pct": bool(dsr_probability(x, sr0) >= 0.95) if np.isfinite(dsr_probability(x, sr0)) else False,
            "candidate_count_raw": len(periodic_sharpes.dropna()),
            "trial_dependence_note": "Raw candidate count used; correlated strategies violate independent-trial assumption, so treat DSR as a diagnostic rather than exact inference.",
        })
    dsr = pd.DataFrame(rows).sort_values("dsr_probability", ascending=False)
    dsr.to_csv(OUT / "deflated_sharpe.csv", index=False)

    detail, pbo = cscv_pbo(active, BLOCKS)
    detail.to_csv(OUT / "pbo_cscv.csv", index=False)
    pbo.to_csv(OUT / "pbo_summary.csv", index=False)

    best = dsr.iloc[0] if len(dsr) else None
    pbo_value = float(pbo["pbo"].iloc[0]) if len(pbo) else np.nan
    status = pd.DataFrame([{
        "best_dsr_strategy": best["strategy"] if best is not None else "",
        "best_dsr_probability": float(best["dsr_probability"]) if best is not None else np.nan,
        "any_dsr_95pct": bool(dsr["passes_dsr_95pct"].any()) if len(dsr) else False,
        "cscv_pbo": pbo_value,
        "pbo_below_0_20": bool(np.isfinite(pbo_value) and pbo_value < 0.20),
        "production_gate": "PASS" if len(dsr) and bool(dsr["passes_dsr_95pct"].any()) and np.isfinite(pbo_value) and pbo_value < 0.20 else "FAIL",
        "note": "Statistical overfit gate only; point-in-time universe, delisted-price coverage, and macro-vintage gates remain separate requirements.",
    }])
    status.to_csv(OUT / "overfit_status.csv", index=False)

    print("\n=== DEFLATED SHARPE ===")
    print(dsr.head(10).to_string(index=False))
    print("\n=== CSCV PBO ===")
    print(pbo.to_string(index=False))
    print("\n=== OVERFIT STATUS ===")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
