# Regime Factor Investing

S&P 500採用銘柄を対象に、**テクニカル、ファンダメンタル、セクターローテーション、金利、信用環境、景気後退リスク**を統合して順位付けする研究用スクリーナーです。

> Research only. Backtest rankings and portfolio weights are not investment advice and do not guarantee future returns.

## Live screener

現在のライブモデルは以下を使用します。

- **Universe**: 実行時点のS&P 500構成銘柄
- **Momentum / Technical**: 12-1M、6M、3M、対SPY相対強度、200日線、50/200日線、52週高値、RSI、実現ボラティリティ
- **Fundamentals**: SEC EDGAR Company FactsからTTM財務を構築
- **Value**: Earnings Yield、FCF Yield、Book-to-Price
- **Quality**: ROA、ROE、営業利益率、営業CFマージン、負債比率、現金比率
- **Growth**: TTM売上高・純利益の前年同期比
- **Sector Rotation**: 11 Select Sector ETFの対SPY相対モメンタム + セクター内部ブレッドス
- **Macro**: Sahm Rule、10Y-3M、HY OAS、NFCI、10年実質金利、FF金利、失業率、CPI、CFNAI
- **Regimes**: `RISK_ON`, `NEUTRAL`, `HIGH_REAL_RATES`, `STAGFLATION`, `RECESSION`
- **Portfolio**: 現在は銘柄・セクターのハード上限を置かず、集中度を別途計測

`config.yaml` の `max_stock_weight` / `max_sector_weight` は現在 `1.0` で非拘束です。旧12%/30%制約版は研究上の比較ベースラインとしてのみ残しています。

## Data sources

| Data | Source | Cost |
|---|---|---:|
| Current S&P 500 constituents | Wikipedia | Free |
| Price / volume | Yahoo Finance via `yfinance` | Free* |
| Company fundamentals | SEC EDGAR Company Facts | Free |
| Macro / rates / credit | FRED | Free |
| Compute | GitHub Actions standard runner on public repo | Free |

\* `yfinance` is unofficial. Terms and endpoint behavior can change.

## Daily GitHub Actions

`.github/workflows/daily-screener.yml` runs weekdays at **07:30 Asia/Tokyo** and commits the current ranking directly to `main`. Pull requests are intentionally not used.

Outputs:

- `data/latest/results.csv`
- `data/latest/portfolio.csv`
- `data/latest/macro_snapshot.csv`
- `data/latest/portfolio_diagnostics.csv`

The diagnostics file records realized concentration such as max stock weight, Top-3 weight, HHI, effective number of holdings, max sector weight and effective sector count.

### SEC User-Agent

Automated SEC requests require an identifiable User-Agent. Add repository secret `SEC_USER_AGENT`, for example:

```text
Your Name your-email@example.com
```

## Research methodology

The repository deliberately separates **screening** from **evidence that the strategy works**.

### Bias-reduced V2

`backtest_v2.py` corrected important V1 problems:

- month-end data are used to compute a signal only after the close;
- execution occurs at the **next trading-session adjusted open**;
- SEC facts must have been filed on/before the signal date;
- monthly macro variables use conservative publication lags;
- trading cost is charged per dollar actually traded.

The completed strict 12% stock / 30% sector V2 baseline for 2018-01 through 2026-07 was approximately:

| Portfolio | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Regime Factor V2 strict baseline | 14.32% | 0.83 | -21.22% |
| SPY | 13.96% | 0.85 | -23.31% |
| RSP | 10.76% | 0.65 | -29.88% |

This is not treated as final alpha evidence because point-in-time membership and macro-vintage biases remain.

### Research matrix

`research_backtest.py` / `research_candidates.py` compare many model components while reusing the same underlying data load:

- inverse-vol, equal-weight and score-proportional allocation;
- Top 5 / 10 / 20 breadth;
- constrained vs unconstrained portfolios;
- dynamic regime weights vs static neutral weights;
- direct macro-sector tilt ablation;
- one-factor and interaction ablations;
- transaction-cost sensitivity;
- subperiod / regime behavior;
- realized concentration.

The strongest **selected in-sample** candidate currently found removes standalone Growth, Low-Vol ranking and direct Macro-Sector tilt. With score-proportional allocation it produced about **24.35% CAGR**, Sharpe **1.10** and max drawdown **-19.49%** over the 102-month research sample. This is explicitly **not a production forecast**: it was found after testing multiple variants and therefore receives a multiple-testing penalty.

### Statistical anti-overfit gates

The repository now includes:

- `research_trial_ledger.csv` — records every strategy trial, including invalidated experiments;
- `research_validation.py` — block bootstrap, rolling windows and split-sample stability;
- `research_overfit.py` — Deflated Sharpe Ratio and CSCV Probability of Backtest Overfitting;
- `research_overfit_sensitivity.py` — sensitivity to assumed trial count and CSCV block count;
- `component_attribution.py` — paired 12-month circular-block bootstrap for the marginal contribution of each component.

A strategy is not promoted merely because it has the highest CAGR. The research workflow runs the matrix and all statistical diagnostics **atomically in the same Actions job**, so DSR/PBO cannot silently remain stale after the matrix changes.

Current component evidence from the bias-reduced sample is strongest for **Value**. Retaining Value versus removing it increased annualized mean return by about **8.57 percentage points**, with a 12-month block-bootstrap 95% interval of approximately **+3.27% to +13.96%**. Momentum and Quality are positive but less statistically decisive. The current standalone Growth score, Low-Vol ranking score and direct Macro-Sector tilt have negative point estimates, but these findings are still treated as hypotheses rather than universal factor conclusions.

## Point-in-time universe research

Current-member backtests have survivorship bias, so several independent reconstructions are audited under `data/research/`.

- `historical_universe_compare.py` — compares public reconstruction datasets;
- `official_sp500_change_audit.py` — checks reconstructions against dated S&P Global constituent-change announcements;
- `pitindex_audit.py` — cross-audits the `pitindex` event-driven free dataset against official changes and the monthly reconstruction;
- `historical_price_coverage.py` — measures whether historical members have usable Yahoo price history;
- `alternate_price_probe.py` — tests alternate free price routes without automatically mixing incompatible price-adjustment conventions.

A curated official S&P spot audit currently found the Pierre month-end reconstruction consistent on all 9 tested add/delete events, while the Hans snapshot series passed 4/9. This is a spot audit, not certification of the entire history.

The historical price problem remains harder: Yahoo retrieves nearly all observations for symbols it successfully transports, but a material group of old/delisted tickers cannot be fetched reliably. A point-in-time universe is therefore not considered production-safe until the delisted-price path is independently validated.

## Macro vintages

The current backtest uses conservative release lags, but most FRED series are still latest-vintage histories. FRED/ALFRED supports historical real-time periods and vintage dates through the official API. A future vintage-correct run should use those endpoints rather than assuming the latest revised history was known in the past.

## Forward out-of-sample protocol

Historical optimization is not allowed to rewrite the forward record. A frozen shadow cohort was registered on **2026-08-21**. Its first genuinely unseen signal is the **2026-08 month-end** signal, executed on the next trading session. Future model changes must form a separately timestamped cohort rather than rewriting the old cohort.

## Research direction

The next priority is not to generate more arbitrary parameter combinations. It is to reduce model and data uncertainty:

1. finish point-in-time constituent validation;
2. obtain a validated delisted-price path;
3. replace revised macro history with ALFRED vintages where feasible;
4. redesign Quality/Growth using academically established definitions rather than assuming one-year revenue/earnings growth is the correct quality-growth measure;
5. separate stock-selection alpha from sector-allocation alpha;
6. accumulate frozen forward observations.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your-email@example.com"
python daily_run.py --top 25 --portfolio 10 --output results.csv
```

Offline tests:

```bash
python test_offline.py
python test_constraints.py
python test_forward_shadow.py
```

## Important limitations

- Rankings are research outputs, not guarantees or investment instructions.
- Yahoo/yfinance can throttle, change endpoints or fail for delisted securities.
- Current-member historical tests retain survivorship bias until the PIT universe is promoted.
- Latest-vintage FRED values can retain revision bias until ALFRED vintages are used.
- Current GICS sector labels are not automatically point-in-time classifications.
- REIT / Financial accounting requires sector-specific treatment; standardized FFO/AFFO is not yet robust.
- Analyst estimates, forward EPS revisions, options, short interest and issuer-level credit are not yet included because reliable point-in-time free data is not established.
- Every added strategy variant increases the multiple-testing burden and must be recorded in the trial ledger.
