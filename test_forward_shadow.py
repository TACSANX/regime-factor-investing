from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import forward_shadow as fs


def test_target_period() -> None:
    d = datetime(2026, 9, 1, 8, 15, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert str(fs.target_period(d)) == "2026-08"


def test_first_build_traded_notional() -> None:
    assert fs.traded_notional({"A": 0.6, "B": 0.4}, None) == 1.0


def test_rebalance_traded_notional() -> None:
    # Sell 20% of A and buy 20% of B => 40% gross traded notional.
    got = fs.traded_notional({"A": 0.4, "B": 0.6}, {"A": 0.6, "B": 0.4})
    assert abs(got - 0.4) < 1e-12


def test_end_weights() -> None:
    w = fs.end_weights({"A": 0.5, "B": 0.5}, {"A": 0.10, "B": -0.10})
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["A"] > w["B"]


def test_empty_csv_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "empty.csv"
        p.write_text("", encoding="utf-8")
        df = fs.load_csv(p, fs.RETURN_COLUMNS)
        assert list(df.columns) == fs.RETURN_COLUMNS
        assert df.empty


def test_date_filled_detection() -> None:
    import pandas as pd

    assert fs._all_dates_filled(pd.Series(["2026-09-01", "2026-09-01"]))
    assert not fs._all_dates_filled(pd.Series([np.nan, np.nan]))
    assert not fs._all_dates_filled(pd.Series(["", ""]))


if __name__ == "__main__":
    test_target_period()
    test_first_build_traded_notional()
    test_rebalance_traded_notional()
    test_end_weights()
    test_empty_csv_round_trip()
    test_date_filled_detection()
    print("forward shadow offline tests: OK")
