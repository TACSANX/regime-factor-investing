import numpy as np
import pandas as pd
from pathlib import Path
import sys, types
sys.modules.setdefault("yfinance", types.SimpleNamespace(download=None))
sys.path.insert(0, str(Path(__file__).parent))
import screener

# TTM extraction: FY 2024=100, Q2 2024 YTD=45, FY2023=90, Q2 2023 YTD=40, Q2 2022=35
records = pd.DataFrame([
    {"start":"2023-01-01","end":"2023-12-31","filed":"2024-02-01","val":90,"form":"10-K","fy":2023,"fp":"FY"},
    {"start":"2024-01-01","end":"2024-12-31","filed":"2025-02-01","val":100,"form":"10-K","fy":2024,"fp":"FY"},
    {"start":"2022-01-01","end":"2022-06-30","filed":"2022-08-01","val":35,"form":"10-Q","fy":2022,"fp":"Q2"},
    {"start":"2023-01-01","end":"2023-06-30","filed":"2023-08-01","val":40,"form":"10-Q","fy":2023,"fp":"Q2"},
    {"start":"2024-01-01","end":"2024-06-30","filed":"2024-08-01","val":45,"form":"10-Q","fy":2024,"fp":"Q2"},
    {"start":"2025-01-01","end":"2025-06-30","filed":"2025-08-01","val":50,"form":"10-Q","fy":2025,"fp":"Q2"},
])
for c in ["start","end","filed"]:
    records[c]=pd.to_datetime(records[c])
records["days"]=(records["end"]-records["start"]).dt.days
cur, prev=screener.ttm_from_records(records)
assert abs(cur - 105) < 1e-9, (cur, prev)
assert abs(prev - 95) < 1e-9, (cur, prev)

# Percentile direction.
s=pd.Series([1,2,3,4,5], dtype=float)
assert screener.percentile_score(s).iloc[-1] == 1.0
assert screener.percentile_score(s, False).iloc[-1] == 0.0

# Portfolio should sum to one and honor stock cap in a feasible case.
df=pd.DataFrame({
    "eligible":[True]*10,
    "score":np.linspace(95,80,10),
    "vol_63d":np.linspace(.15,.30,10),
    "GICS Sector":["A","A","B","B","C","C","D","D","E","E"],
    "Symbol":[f"T{i}" for i in range(10)],
    "Security":[f"Co{i}" for i in range(10)],
})
p=screener.build_portfolio(df,10,.12,.30)
assert abs(p["weight"].sum()-1) < 1e-8
assert p["weight"].max() <= .120001
assert p.groupby("GICS Sector")["weight"].sum().max() <= .300001
print('offline tests passed')
