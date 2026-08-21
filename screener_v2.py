from __future__ import annotations

import screener as sc
from portfolio_constraints import build_portfolio_strict


# Legacy/comparison entry point only.  The live daily strategy is intentionally
# uncapped and runs through daily_run.py.  Keep this wrapper pinned to the old
# 12% single-stock / 30% sector constraints so config.yaml's non-binding 1.0
# values cannot silently change the meaning of the strict research baseline.
def build_legacy_strict(ranked, n, _max_stock, _max_sector):
    return build_portfolio_strict(ranked, n, 0.12, 0.30)


sc.build_portfolio = build_legacy_strict

if __name__ == "__main__":
    sc.main()
