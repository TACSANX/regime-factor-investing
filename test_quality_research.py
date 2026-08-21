import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("yfinance", types.SimpleNamespace(download=None))
sys.path.insert(0, str(Path(__file__).parent))

import quality_research as qr


# Pick the instant fact closest to one year before the latest, not the previous
# quarter. This is required for one-year quality-growth comparisons.
instant = pd.DataFrame([
    {"end": "2024-03-31", "filed": "2024-05-01", "val": 80.0, "form": "10-Q"},
    {"end": "2024-06-30", "filed": "2024-08-01", "val": 90.0, "form": "10-Q"},
    {"end": "2024-09-30", "filed": "2024-11-01", "val": 95.0, "form": "10-Q"},
    {"end": "2024-12-31", "filed": "2025-02-01", "val": 100.0, "form": "10-K"},
    {"end": "2025-03-31", "filed": "2025-05-01", "val": 105.0, "form": "10-Q"},
    {"end": "2025-06-30", "filed": "2025-08-01", "val": 120.0, "form": "10-Q"},
])
for col in ["end", "filed"]:
    instant[col] = pd.to_datetime(instant[col])
current, prior = qr.latest_and_prior_instant(instant)
assert current == 120.0
assert prior == 90.0, (current, prior)


# Monotone toy cross-section: profitability/growth improve while leverage,
# beta and idiosyncratic volatility decline.
n = 12
x = np.arange(n, dtype=float)
df = pd.DataFrame({
    "GICS Sector": ["Industrials"] * n,
    "assets": np.full(n, 100.0),
    "assets_prior": np.full(n, 95.0),
    "equity": np.full(n, 50.0),
    "equity_prior": np.full(n, 47.0),
    "revenue_ttm": 100.0 + x * 3.0,
    "revenue_prior_ttm": 95.0 + x * 2.0,
    "gross_profit_ttm": 20.0 + x * 2.0,
    "gross_profit_prior_ttm": 18.0 + x * 1.0,
    "net_income_ttm": 4.0 + x * 0.8,
    "net_income_prior_ttm": 3.5 + x * 0.4,
    "cfo_ttm": 6.0 + x * 1.0,
    "cfo_prior_ttm": 5.0 + x * 0.5,
    "debt": 60.0 - x * 3.0,
    "beta_252": 1.5 - x * 0.06,
    "idio_vol_252": 0.45 - x * 0.02,
})
out = qr.add_qmj_proxy_features(df)
assert out["quality_qmj_proxy"].notna().all()
assert out.loc[n - 1, "quality_qmj_proxy"] > out.loc[0, "quality_qmj_proxy"]
assert out["quality_qmj_component_count"].min() == 3
print("quality research tests passed")
