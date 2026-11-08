# 📈 Nifty 500 Relative Stock Valuation & ML Screener

An end-to-end Machine Learning pipeline and interactive web application designed to evaluate Indian equities (Nifty 500 index) using relative fundamental valuation models. 

The project automates data extraction from NSE and Yahoo Finance, engineers robust financial features, trains a Random Forest regression model with strict cross-validation, and serves predictions via a **Streamlit Cloud** web interface.

---

## 🚀 Project Architecture & File Structure

The project is structured into sequential steps and modular components:

* **`valuation_utils.py`**: Core utility module containing custom scikit-learn transformers (`Winsorizer`, `SectorMedianRegressor`), consistent feature engineering logic, and financial metric extractors.
* **`step_1_data_pipeline.ipynb`**: Downloads the Nifty 500 ticker list from NSE, fetches fundamental and growth/risk metrics via `yfinance`, cleans the dataset, and exports `nifty500_clean_v2.csv`.
* **`step_2_train_model.ipynb`**: Trains a Random Forest Regressor (with target log-transformation), compares it against a sector-median baseline using repeated K-Fold cross-validation, computes out-of-fold empirical confidence intervals, and serializes the model artifact (`nifty_valuation_model.pkl`).
* **`step_3_evaluate.ipynb`**: Batch screening and single-stock evaluation script used for offline validation.
* **`app.py`**: Interactive Streamlit web application for real-time stock evaluation and screening.

---

## 📊 Model Performance & Insights

The model predicts **Trailing P/E** multiples based on profitability ratios (ROE, ROA, Margins), growth/risk metrics (Revenue/Earnings CAGR, Earnings CV, Beta), and sector categories.

### Cross-Validation Results (Repeated K-Fold)
* **Sector Median Baseline:** MAE: `25.77` | MedAE: `13.57` | $R^2$: `-0.027`
* **Random Forest Regressor:** MAE: `22.47` | MedAE: `12.28` | $R^2$: `0.207 (±0.075)`
* **Benchmark Comparison:** The Random Forest model outperforms the traditional Sector Median benchmark (measured by Median Absolute Error) in **80% of cross-validation folds**.
* **Uncertainty Bounds:** Out-of-fold residual analysis establishes an **80% P/E interval multiplier** of `x0.48` to `x2.19` around the Fair P/E.

### Sample Batch Screener Output
| Ticker | Sector | Price (₹) | Actual P/E | Fair P/E | P/E Min | P/E Max | P/E / Fair | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INFY.NS** | Technology | 1,171.60 | 15.60 | 29.52 | 14.22 | 64.53 | 0.53 | Within range |
| **SBIN.NS** | Financial Services | 1,016.00 | 10.37 | 13.95 | 6.72 | 30.49 | 0.74 | Within range |
| **RELIANCE.NS** | Energy | 1,324.00 | 21.82 | 16.72 | 8.05 | 36.55 | 1.30 | Within range |
| **HINDUNILVR.NS** | Consumer Defensive | 1,997.00 | 44.34 | 25.91 | 12.47 | 56.62 | 1.71 | Within range |

> *Note on Model Limitations:* While fundamental features capture a baseline structural valuation, market multiples are heavily influenced by forward-looking expectations, macro sentiment, and liquidity flows that historical financial statements alone cannot fully explain. This tool serves as a quantitative screening baseline rather than a standalone financial advisor.

---

## 🌐 Live Web App
Explore the interactive dashboard live on Streamlit Cloud:  
👉 [Nifty 500 Valuation Dashboard](https://nifty-stock-valuation-f2apznurlbbnnscwce8bcs.streamlit.app/)