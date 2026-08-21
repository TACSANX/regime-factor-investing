# Regime Factor Investing

S&P 500 constituentsを対象に、**テクニカル、ファンダメンタル、セクターローテーション、金利、信用環境、景気後退リスク**を統合し、マクロレジームに応じて因子ウェイトを変える日次スクリーナーです。

> This is a research tool, not investment advice. Rankings are model outputs and do not guarantee future returns.

## Overview

このモデルは「すべての因子を固定ウェイトで足す」のではなく、マクロ環境を制御層として使います。

- **Universe**: 実行時点のS&P 500構成銘柄
- **Momentum / Technical**: 12-1M、6M、3M、対SPY相対強度、200日線、50/200日線、52週高値、RSI、実現ボラティリティ
- **Fundamentals**: SEC EDGAR Company FactsからTTM財務データを構築
- **Value**: Earnings Yield、FCF Yield、Book-to-Price
- **Quality**: ROA、ROE、営業利益率、営業CFマージン、負債比率、現金比率
- **Growth**: TTM売上高・純利益の前年同期比
- **Sector Rotation**: 11 Select Sector ETFの対SPY相対モメンタム + セクター内部ブレッドス
- **Macro Regime**: Sahm Rule、10Y-3M、HY OAS、NFCI、10年実質金利、FF金利、失業率、CPI、CFNAI
- **Regimes**: `RISK_ON`, `NEUTRAL`, `HIGH_REAL_RATES`, `STAGFLATION`, `RECESSION`
- **Portfolio Construction**: score × inverse volatility。現在のライブ設定では銘柄・セクターのハード上限を設けず、集中度を計測して評価します。

## Data sources

| Data | Source | Cost |
|---|---|---:|
| S&P 500 constituents | Wikipedia | Free |
| Price / volume | Yahoo Finance via `yfinance` | Free* |
| Company fundamentals | SEC EDGAR Company Facts | Free |
| Macro / rates / credit | FRED CSV endpoints | Free |
| Compute | GitHub Actions standard runner on public repo | Free |

\* `yfinance` is an unofficial client. Review Yahoo/yfinance terms before non-personal or commercial use.

## GitHub Actions: zero-cost daily operation

`.github/workflows/daily-screener.yml` runs automatically at **07:30 Asia/Tokyo, Monday-Friday** and:

1. installs dependencies;
2. restores the screener cache;
3. runs offline tests;
4. downloads market, SEC and FRED data;
5. ranks S&P 500 constituents;
6. uploads a 14-day Actions artifact;
7. commits the latest outputs directly to `main` under `data/latest/`.

Canonical outputs:

- `data/latest/results.csv` — full ranking
- `data/latest/portfolio.csv` — reference portfolio
- `data/latest/macro_snapshot.csv` — macro values used in the run

The workflow intentionally does **not** create pull requests.

### One-time setup: SEC User-Agent

SEC requires automated requests to declare an identifiable User-Agent. Add one repository secret:

1. Open **Settings → Secrets and variables → Actions**.
2. Create a repository secret named `SEC_USER_AGENT`.
3. Use a value such as:

```text
Your Name your-email@example.com
```

The workflow fails fast if this secret is missing.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your-email@example.com"
```

Run:

```bash
python screener.py --top 25 --portfolio 10 --output results.csv
```

Offline tests:

```bash
python test_offline.py
```

## Configuration

`config.yaml` controls cache TTLs, factor weights and regime-specific sector tilts. `max_stock_weight` / `max_sector_weight` are currently `1.0`, so they are non-binding in the live reference portfolio.

Current factor-weight sets:

| Regime | Main bias |
|---|---|
| `RISK_ON` | Momentum / Growth / cyclical rotation |
| `NEUTRAL` | Balanced multi-factor |
| `HIGH_REAL_RATES` | Quality / Value, lower duration exposure |
| `STAGFLATION` | Quality / Value / inflation-sensitive sectors |
| `RECESSION` | Quality / Low Vol / defensives |

## Backtest research

The repository separates headline screening from research validation.

### Bias-reduced V2 baseline

`backtest_v2.py` fixes two important V1 problems:

- signal is computed using month-end data, but execution occurs at the **next trading-session adjusted open**;
- monthly macro series use conservative publication lags and SEC fundamentals are restricted to facts filed on/before the signal date.

The completed V2 run (2018-01 through 2026-07, 10 bps per dollar traded) produced approximately:

| Portfolio | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Regime Factor V2, strict 12% / 30% baseline | 14.32% | 0.83 | -21.22% |
| SPY | 13.96% | 0.85 | -23.31% |
| RSP | 10.76% | 0.65 | -29.88% |

This result is **not** considered final evidence of alpha because current S&P 500 membership and latest-vintage FRED histories still create survivorship/revision bias.

### Research matrix

`research_backtest.py` reuses one data load to compare:

- uncapped inverse-volatility × score allocation (current base research strategy);
- equal-weight and score-only allocation;
- Top 5 / 10 / 20 breadth;
- the old strict 12% stock / 30% sector baseline;
- static neutral weights vs dynamic macro-regime weights;
- removal of macro-sector tilt;
- one-factor-at-a-time ablations for momentum, quality, value, growth, low-volatility and sector rotation;
- transaction costs from 0 to 100 bps per dollar traded;
- subperiod and regime-level results;
- realized concentration: max stock/sector weight, HHI and effective number of holdings/sectors.

Outputs are written to `data/research_backtest/` by `.github/workflows/research-backtest.yml`.

### Historical-universe audit

The repository also tests whether a survivorship-reduced backtest can be built with free data:

- `historical_universe_compare.py` compares two public historical S&P 500 reconstruction datasets;
- `historical_price_coverage.py` measures whether Yahoo has usable historical prices for those members and retries failed bulk downloads individually;
- results are stored under `data/research/`.

Historical constituent sources still disagree on ticker history and corporate renames, so they are not automatically promoted to the production backtest until identity normalization and price coverage pass quality thresholds.

## Important limitations

- **No return guarantee**: a ranking is not a buy/sell instruction.
- **Free-market-data reliability**: Yahoo endpoints may change, throttle or fail.
- **No analyst-estimate layer**: forward EPS revisions, estimate dispersion and high-quality forward valuation are intentionally excluded because consistent free point-in-time data is difficult to obtain.
- **REIT / Financial accounting differences**: sector-specific logic is used, but REIT FFO/AFFO is not robustly standardized in this version.
- **Backtest bias**: current-member backtests still have S&P 500 survivorship bias; latest-vintage FRED data can contain revision bias.
- **SEC data timing**: Company Facts updates when filings become available; fundamentals are not real-time market data.
- **Research selection bias**: factor ablations are diagnostics. Choosing the best in-sample variant and declaring it production-ready would be data mining; confirmation must use a separate out-of-sample protocol.

## Project files

```text
.
├── .github/workflows/
│   ├── daily-screener.yml
│   ├── backtest.yml
│   ├── backtest-v2-once.yml
│   ├── research-backtest.yml
│   ├── historical-universe-compare.yml
│   └── historical-price-coverage.yml
├── backtest.py
├── backtest_v2.py
├── research_backtest.py
├── historical_universe_compare.py
├── historical_price_coverage.py
├── portfolio_constraints.py
├── config.yaml
├── requirements.txt
├── screener.py
├── test_offline.py
├── data/backtest/
├── data/backtest_v2/
├── data/research/
└── README.md
```

## Possible extensions

With a suitable licensed data source, useful additions include:

- EPS revision breadth / dispersion
- Forward P/E and forward FCF yield
- Earnings surprise history
- Options IV / skew / term structure
- Short interest / securities lending
- Insider and institutional ownership changes
- Point-in-time index membership with verified corporate-identity history
- Vintage macro data (for example, ALFRED-style vintages)
- Issuer-level bond spreads / CDS
