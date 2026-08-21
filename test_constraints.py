import numpy as np
import pandas as pd

from portfolio_constraints import build_portfolio_strict

# Deliberately put the highest-ranked names into only two sectors.  The old
# implementation selected these first and later violated the 30% sector cap.
rows = []
for i in range(30):
    if i < 8:
        sec = "Tech"
    elif i < 16:
        sec = "Health"
    elif i < 20:
        sec = "Financials"
    elif i < 24:
        sec = "Industrials"
    elif i < 27:
        sec = "Energy"
    else:
        sec = "Utilities"
    rows.append({
        "eligible": True,
        "score": 100 - i,
        "vol_63d": 0.15 + i * 0.002,
        "GICS Sector": sec,
        "Symbol": f"T{i}",
        "Security": f"Company {i}",
    })
ranked = pd.DataFrame(rows)

p = build_portfolio_strict(ranked, 10, 0.12, 0.30)
assert len(p) == 10
assert abs(float(p["weight"].sum()) - 1.0) < 1e-8
assert float(p["weight"].max()) <= 0.1200001
assert float(p.groupby("GICS Sector")["weight"].sum().max()) <= 0.3000001
assert p["GICS Sector"].nunique() >= 4

print("strict portfolio constraint tests passed")
