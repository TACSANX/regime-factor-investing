from __future__ import annotations

import screener as sc
from portfolio_constraints import build_portfolio_strict

# Reuse the data/factor engine but replace the portfolio constructor with a
# constraint-safe implementation.
sc.build_portfolio = build_portfolio_strict

if __name__ == "__main__":
    sc.main()
