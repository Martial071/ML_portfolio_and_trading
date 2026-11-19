# News-Driven Commodity Portfolio

**Can LLM-scored news and macro data predict weekly commodity futures returns, and does a cost-aware optimiser turn those forecasts into a better portfolio?**

Course project for *Machine Learning for Portfolio Management and Trading* (IP Paris), built on the course library `skfin`.

**Authors:** Goueudeupouh Diomande · Jude Nemer Khazzaka

---

## Overview

| | |
|---|---|
| **Universe** | Five front-month futures: WTI crude oil, natural gas, gold, copper, wheat (Yahoo Finance `CL=F`, `NG=F`, `GC=F`, `HG=F`, `ZW=F`) |
| **Frequency** | Weekly. Weeks run Saturday–Friday and are labelled by their Friday. 2010–2026, 873 weeks × 5 commodities |
| **Text** | Nine news and speech sources, scored article by article with `gpt-4o-mini` (structured output: impact in [−1, 1] on each commodity over the next week) |
| **Macro** | Six FRED series (rates, yield curve, dollar, VIX, breakeven inflation), lagged one week |
| **Target** | Next week's return $r_{i,t+1}$, predicted from information available at the Friday close of week $t$ |
| **Models** | Ridge (baseline), Random Forest, LightGBM. One model per commodity |
| **Feature sets** | Nested: **A** price ⊂ **B** price + news ⊂ **C** price + news + macro |
| **Portfolio** | Mean-variance with quadratic transaction costs (Gârleanu & Pedersen, 2013): the **dynamic** optimiser takes previous holdings into account and is compared with the **myopic** one |
| **Evaluation** | Train 2010–2018 · validation 2019–2022 · test 2023-01 → 2026-09. The specification is frozen on validation before the test is run, once |

## Hypotheses

| | Hypothesis |
|---|---|
| H1 | News features add predictability beyond prices |
| H2 | Macro features add predictability beyond prices and news |
| H3 | The dynamic (cost-aware) optimiser beats the myopic one, net of costs |
| H4 | The result survives the out-of-sample test, sub-periods and multiple testing |

## Results

**Frozen specification:** Random Forest, feature set C, dynamic optimiser.
**Net Sharpe 0.71 on validation → 0.60 on the 2023–2026 test** (0.53 with the myopic optimiser), but $t \approx 1.06$, and 55 % of the test PnL comes from 2023.

**Validation net Sharpe (2019–2022, dynamic optimiser)**

| Model | A: price | B: + news | C: + macro |
|---|---|---|---|
| Ridge | **0.85** | −0.06 | 0.18 |
| Random Forest | 0.43 | 0.56 | 0.72 |
| LightGBM | 0.26 | 0.25 | 0.73 |

Ridge A has the best validation score but was not selected: it collapses as soon as features are added and rests largely on a single crude-oil episode (spring 2020). Random Forest C and LightGBM C are statistically indistinguishable (Sharpe standard error ≈ 0.5); Random Forest was preferred because it has fewer interacting hyperparameters.

**Verdicts**

| Hypothesis | Validation (2019–2022) | Test (2023–2026) | Verdict |
|---|---|---|---|
| H1 — news | Net Sharpe B − A: +0.13 (RF), −0.004 (LightGBM), −0.90 (Ridge) | not tested separately | **Not supported** |
| H2 — macro | Net Sharpe C − B: +0.48 (LightGBM), +0.16 (RF), +0.23 (Ridge) | RF C: 0.595, IC > 0 on 4/5 commodities | **Supported in validation**, not significant |
| H3 — dynamic optimiser | Higher net Sharpe in 7/9 configurations; gap grows with the cost level | 0.595 (dynamic) vs 0.531 (myopic) | **Supported in direction**, small at the calibrated cost |
| H4 — robustness | ~30 configurations tried | $t \approx 1.06$; 55 % of PnL from 2023 | **Not demonstrated statistically** |

**Main findings**
- Weekly commodity returns are close to unpredictable. LLM-scored news adds little on its own, and news–return correlations change sign between periods half of the time.
- Macro conditions are the most useful addition, mainly for shallow tree models: with ~460 training weeks per commodity, a linear model cannot exploit 14 noisy features.
- The dynamic optimiser behaves as theory predicts: identical to the myopic one at zero cost, increasingly better as costs rise, with longer holding periods.
- Test-period information coefficients are positive for four commodities out of five, but only natural gas (0.147) clears the noise band (±0.144). The portfolio gain comes from many small edges, which is a thin base with five assets (Sharpe ≈ IC × √breadth).

## Pipeline

```
9 raw text files ──► articles.csv ──► LLM scores (scores_*.csv) ──► weekly_news.csv ──┐
Yahoo Finance futures ──► futures_daily_close.csv ──► weekly price features ──────────┴─► panel_weekly.csv ──┐
FRED (6 series) ──► macro_daily_raw.csv ──► macro_weekly.csv (lagged 1 week) ────────────────────────────────┴─► panel_weekly_plus_macro.csv
```

The notebook follows the project steps: question and hypotheses → text data scored with an LLM → cleaning → exploratory analysis → features and target → return prediction (three models × three feature sets) → dynamic portfolio optimisation with transaction costs → backtest → results and attribution → robustness → out-of-sample test and conclusion.

## Features

| Block | Columns | Information used |
|---|---|---|
| Price | `mom_4w`, `mom_12w` (momentum), `vol_12w` (12-week volatility) | closes up to Friday $t$ |
| News | `net_tone` (mean LLM impact of relevant articles), `n_relevant`, `has_news`, `attention` = log(1 + n) | articles dated in week $t$ |
| Macro | fed funds rate (`DFF`), 10y yield (`DGS10`), 10y−2y slope (`DGS10` − `DGS2`), 4-week dollar change (`DTWEXBGS`), VIX (`VIXCLS`), 10y breakeven (`T10YIE`) | FRED, up to Friday $t-1$ |
| Target | `target` = $r_{i,t+1}$, never used as a feature | — |

Weekly returns are capped at ±50 % to remove contract-roll artefacts in the continuous futures series.

## Data

**Text sources** (raw files, never modified):

| File | Text used |
|---|---|
| `all_speeches_90_22.csv` | full speech |
| `communications_20_27.csv` | communication text |
| `CrudeOilNewsData_26.csv` | article description |
| `geopolitical_events_timeline.csv` | event description |
| `gold-dataset-sinha-khandait_00_19.csv` | gold headlines (Sinha & Khandait) |
| `guardian_environment_news_17_24.csv` | article text |
| `Newsdata_Records_2026_08_27_08_18_00.csv` | AI summary |
| `nyt_2022_2025.csv` | lead paragraph |
| `sp500_headlines_2008_2024.csv` | headline |

**Prices:** Yahoo Finance via `yfinance`. **Macro:** FRED public CSV endpoint (no API key needed).

**What is in the repository.** The raw news archive is too large for GitHub and is not tracked. It is hosted on Google Drive and downloaded automatically by the notebook when `Download_Raw_News = True`. The processed files the notebook needs (`weekly_news.csv`, `futures_daily_close.csv`, the `macro/` folder, `panel_weekly.csv`, `panel_weekly_plus_macro.csv`) are included, so the full analysis runs without downloading or scoring anything.

## How to run

1. **Environment.** Use the course's conda environment with `skfin` installed, plus:
   ```bash
   conda install -c conda-forge lightgbm yfinance pydantic openai
   pip install gdown py7zr   # only needed to download the raw news archive
   ```
   On Windows, install LightGBM from conda-forge rather than pip, so its OpenMP runtime matches the rest of the environment.
2. **Open the notebook and run all cells.** With the default switches (all `False`), it reads the saved files: no download, no API call.
3. **Switches** (in the settings cell):

   | Switch | `True` means | Cost |
   |---|---|---|
   | `Download_Raw_News` | download and unpack the raw news archive from Google Drive | seconds |
   | `REBUILD_NEWS` | rebuild articles → LLM scores → `weekly_news.csv`; only articles without a saved score are sent to the API | paid, ≈ 40 min for a full run |
   | `DOWNLOAD_PRICES` | download futures prices again (otherwise read from file, so Yahoo revisions do not change results) | seconds |
   | `DOWNLOAD_MACRO` | download the FRED series again | seconds |

4. **OpenAI key** (only for `REBUILD_NEWS = True`). Store it in the conda environment, never in the notebook:
   ```bash
   conda env config vars set OPENAI_API_KEY=sk-...
   conda deactivate && conda activate <env_name>
   ```

To add a news source: put the raw file in `RAW_DIR`, add it to `RAW_FILES` with its date and text columns, and run once with `REBUILD_NEWS = True`. Only the new articles are scored. To extend the period, change `END` and set `DOWNLOAD_PRICES = True`.

## Limitations

- **LLM look-ahead.** `gpt-4o-mini` was trained on data up to October 2023 and may know how earlier events turned out. The test PnL is reported year by year so that 2024–2026 can be read as the uncontaminated part.
- **News coverage** differs across sources and years, with a hole in 2024–2025, so the test period says more about prices and macro than about news.
- **Publication time.** Articles are assigned to weeks by date only, not by timestamp.
- **Costs.** Quadratic costs (more conservative than the square-root law), liquidity proxied by volatility in the absence of volume data.
- **Contract rolls.** Front-month series are not roll-adjusted (returns are capped instead).
- **Macro vintages.** FRED serves today's vintage; harmless for market series such as rates, the dollar and the VIX.
- **Breadth.** Five commodities give very few independent bets: the universe, more than the length of the history, is the binding constraint.

## References

- Tetlock (2007), *Giving content to investor sentiment*, Journal of Finance.
- Brandt & Gao (2019), *Macro fundamentals or geopolitical events? A textual analysis of news events for crude oil*, Journal of Empirical Finance.
- Lopez-Lira & Tang (2023), *Can ChatGPT forecast stock price movements?*
- Glasserman & Lin (2023), *Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis*.
- Moskowitz, Ooi & Pedersen (2012), *Time series momentum*, Journal of Financial Economics.
- Gu, Kelly & Xiu (2020), *Empirical asset pricing via machine learning*, Review of Financial Studies.
- Campbell & Thompson (2008), *Predicting excess stock returns out of sample*, Review of Financial Studies.
- Breiman (2001), *Random forests*, Machine Learning; Ke et al. (2017), *LightGBM*, NeurIPS.
- Gârleanu & Pedersen (2013), *Dynamic trading with predictable returns and transaction costs*, Journal of Finance.
- Frazzini, Israel & Moskowitz (2018), *Trading costs*.
- Lo (2002), *The statistics of Sharpe ratios*, Financial Analysts Journal; Harvey, Liu & Zhu (2016), *… and the cross-section of expected returns*, Review of Financial Studies.
- Grinold & Kahn (2000), *Active Portfolio Management*.

**Software:** `skfin` (backtesters, mean-variance estimators, `llm.apply_prompt`), scikit-learn, LightGBM, OpenAI API, `yfinance`, pandas, pydantic.