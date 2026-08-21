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

This is not treated as final alpha evidence because historical-universe and macro-vintage biases remain.

## Current simplified research result

The strongest raw selected in-sample result remains the dynamic score-proportional candidate at about **24.35% CAGR**, but additional tests show that the extra allocation and macro complexity are not statistically distinguishable from simpler alternatives.

The current **primary research candidate** is therefore:

- Value + Momentum + Quality + Sector Rotation;
- fixed `NEUTRAL` factor weights;
- standalone Growth removed;
- Low-Vol ranking removed;
- direct Macro-Sector tilt removed;
- Top 10;
- **equal weight (10% each)**;
- 10 bp cost per dollar traded.

Its 102-month reconstructed result is approximately:

| Portfolio | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| **Static Neutral Equal — primary simple** | **23.21%** | **1.03** | **-20.78%** |
| Dynamic Regime Equal — timing control | 24.25% | 1.10 | -19.44% |
| Dynamic Regime Score — allocation control | 24.35% | 1.10 | -19.49% |
| SPY | 13.96% | 0.85 | -23.31% |

Why prefer the lower-CAGR static model as the research primary?

- Dynamic Equal minus Static Neutral Equal is only about **+0.65 percentage points/year** in mean return.
- Its 12-month block-bootstrap 95% interval is roughly **-2.13% to +3.81%**.
- The probability that Dynamic is actually better in that bootstrap is only about **65%**.
- Score-proportional minus Equal Weight is only about **+0.10 percentage points/year**, also with a confidence interval crossing zero.

The simpler model therefore gets the benefit of the doubt until forward evidence proves that timing or score sizing adds value.

### Robustness of the simple candidate

For Static Neutral Equal:

- block-bootstrap probability of positive SPY active return: about **99.77%**;
- 36-month rolling SPY outperformance rate: **100%**;
- 60-month rolling SPY outperformance rate: **100%**;
- first-half CAGR: about **22.24%**;
- second-half CAGR: about **24.18%**;
- removing each full calendar year from 2018 through 2025 still leaves SPY outperformance in **8/8** tests;
- worst leave-one-year-out CAGR advantage versus SPY is about **+6.80 percentage points/year**.

This is still an in-sample research result, not a forecast.

### What the factor tests currently say

Evidence is strongest for **Value**. In the paired component study, retaining Value versus removing it increased annualized mean return by about **8.57 percentage points**, with a 12-month block-bootstrap 95% interval of approximately **+3.27% to +13.96%**.

Momentum and Quality are positive but less decisive. The current standalone Growth score, Low-Vol ranking score and direct Macro-Sector tilt have negative point estimates, so they are excluded from the simplified research candidate.

Sector Rotation is still retained, but its marginal contribution under the final simplified specification has not yet been isolated cleanly.

## Statistical anti-overfit gates

The repository includes:

- `research_trial_ledger.csv` — records strategy trials, including invalidated experiments;
- `research_validation.py` — block bootstrap, rolling windows and split-sample stability;
- `research_overfit.py` — Deflated Sharpe Ratio and CSCV Probability of Backtest Overfitting;
- `research_overfit_sensitivity.py` — sensitivity to assumed trial count and CSCV block count;
- `component_attribution.py` — paired circular-block bootstrap for marginal component contribution;
- `research_simple_report.py` — lightweight market-state, cost, allocation and leave-one-year-out diagnostics.

A strategy is not promoted merely because it has the highest CAGR. The historical research result remains **research-only** until forward evidence accumulates.

## Turnover and concentration

The simplified Equal Weight Top-10 strategy retains an average of about **5.79 names** from one month to the next, so about **4.21 names are replaced each month**. The next turnover experiment is deliberately limited to one rule: buy Top 10, retain existing holdings while their rank remains within Top 15, and fill vacancies from the highest-ranked names.

The most frequently held historical names include DECK, ANET, NVDA, LRCX and KLAC. Information Technology accounts for roughly 26.6% of holding observations, with meaningful exposure also to Energy, Financials, Consumer Discretionary and Health Care.

## Point-in-time data: pragmatic treatment

Historical membership and delisted-price handling are useful but are no longer the main research bottleneck. The repository keeps lightweight audits and documents the residual bias rather than attempting to reproduce every corporate action perfectly.

A larger official S&P spot audit found the Pierre month-end reconstruction consistent on **30/30 tested add/delete events**. Yahoo remains imperfect for old/delisted symbols. These limitations are disclosed rather than treated as solved.

FRED histories are still largely latest-vintage with conservative release lags. ALFRED vintage data remains a possible later improvement, not a prerequisite for continuing the simpler strategy research.

## Forward out-of-sample protocol

Historical optimization is not allowed to rewrite the forward record.

- **r1** preserves the original broad frozen cohort created on 2026-08-21.
- **r2-simple** freezes the new simplified comparison separately: `Static Neutral Equal` versus `Dynamic Regime Equal`.
- Both start with the first genuinely unseen **2026-08 month-end** signal and next-session execution.
- Later model-definition changes must create another cohort rather than rewriting r1/r2.

No strategy is automatically promoted from a forward result.

## Research direction

Research is intentionally narrow now:

1. finish the already preregistered QMJ-inspired Quality test;
2. test one Top-15 holding buffer to reduce turnover;
3. accumulate r2 forward evidence for Static Neutral Equal versus Dynamic Regime Equal;
4. do **not** add another timing model unless forward evidence gives a clear reason.

The objective is no longer to maximize backtest CAGR by adding parameters. It is to keep the simplest model whose performance remains robust.

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
- Current-member historical tests retain survivorship bias until a PIT universe is used end-to-end.
- Latest-vintage FRED values can retain revision bias until ALFRED vintages are used.
- Current GICS sector labels are not automatically point-in-time classifications.
- REIT / Financial accounting requires sector-specific treatment; standardized FFO/AFFO is not yet robust.
- Every added strategy variant increases the multiple-testing burden and must be recorded in the trial ledger.
