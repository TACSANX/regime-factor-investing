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
- **Portfolio Construction**: score × inverse volatility、銘柄上限・セクター上限

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

SEC currently asks automated users to keep aggregate access at or below 10 requests/second. The implementation sleeps between uncached Company Facts requests and caches responses.

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

`config.yaml` controls cache TTLs, portfolio caps, factor weights and regime-specific sector tilts.

Current factor-weight sets:

| Regime | Main bias |
|---|---|
| `RISK_ON` | Momentum / Growth / cyclical rotation |
| `NEUTRAL` | Balanced multi-factor |
| `HIGH_REAL_RATES` | Quality / Value, lower duration exposure |
| `STAGFLATION` | Quality / Value / inflation-sensitive sectors |
| `RECESSION` | Quality / Low Vol / defensives |

## Important limitations

- **No return guarantee**: a ranking is not a buy/sell instruction.
- **Free-market-data reliability**: Yahoo endpoints may change, throttle or fail.
- **No analyst-estimate layer**: forward EPS revisions, estimate dispersion and high-quality forward valuation are intentionally excluded because consistent free point-in-time data is difficult to obtain.
- **REIT / Financial accounting differences**: sector-specific logic is used, but REIT FFO/AFFO is not robustly standardized in this version.
- **Backtest bias**: using today's S&P 500 membership or revised FRED data for historical tests introduces survivorship / look-ahead bias. Rigorous backtests require point-in-time constituents and vintage macro data.
- **SEC data timing**: Company Facts updates when filings become available; fundamentals are not real-time market data.

## Project files

```text
.
├── .github/workflows/daily-screener.yml
├── config.yaml
├── requirements.txt
├── screener.py
├── test_offline.py
├── data/latest/                  # created by GitHub Actions
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
- Point-in-time index membership
- Issuer-level bond spreads / CDS
- Walk-forward backtesting and transaction-cost modeling
