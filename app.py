import pickle
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from valuation_utils import build_features, fetch_growth_features, FEATURES

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Nifty 500 Stock Valuation Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------------------------------------------
# Load Model Artifacts
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts(model_file="nifty_valuation_model.pkl"):
    try:
        with open(model_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

saved_artifacts = load_model_artifacts()

# ----------------------------------------------------------------------
# Helper Evaluation Function
# ----------------------------------------------------------------------
def evaluate_single_stock(ticker_symbol: str, artifacts: dict):
    info = yf.Ticker(ticker_symbol).info
    pe = info.get('trailingPE')
    pe = float(pe) if pe is not None else np.nan
    price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))

    if pd.isna(price) or pd.isna(pe) or pe <= 0:
        return None, "Invalid price or Trailing P/E data."

    row = {
        'Sector': info.get('sector'),
        'ROE': info.get('returnOnEquity', np.nan),
        'ROA': info.get('returnOnAssets', np.nan),
        'Operating_Margin': info.get('operatingMargins', np.nan),
        'Net_Profit_Margin': info.get('profitMargins', np.nan),
        'Revenue_Growth': info.get('revenueGrowth', np.nan),
        'Earnings_Growth': info.get('earningsGrowth', np.nan),
        'Debt_To_Equity': info.get('debtToEquity', np.nan),
        'Current_Ratio': info.get('currentRatio', np.nan),
        'Free_Cash_Flow': info.get('freeCashflow', np.nan),
        'Total_Revenue': info.get('totalRevenue', np.nan)
    }
    row.update(fetch_growth_features(ticker_symbol, info=info))

    X = build_features(pd.DataFrame([row]))[artifacts['features']]
    fair_pe = float(artifacts['model'].predict(X)[0])
    lo_pe = fair_pe * np.exp(artifacts['q_lo'])
    hi_pe = fair_pe * np.exp(artifacts['q_hi'])

    price_lo = price * lo_pe / pe
    price_hi = price * hi_pe / pe
    fair_price = price * fair_pe / pe

    if pe < lo_pe:
        verdict = "Below Peer Range (Potential Discount)"
        status = "success"
    elif pe > hi_pe:
        verdict = "Above Peer Range (Potential Premium)"
        status = "warning"
    else:
        verdict = "Within Peer Range (Fairly Valued)"
        status = "info"

    results = {
        'Company_Name': info.get('shortName', info.get('longName', ticker_symbol)),
        'Sector': row['Sector'],
        'Current_Price': price,
        'Actual_PE': pe,
        'Fair_PE': fair_pe,
        'PE_Min': lo_pe,
        'PE_Max': hi_pe,
        'Fair_Price': fair_price,
        'Price_Min': price_lo,
        'Price_Max': price_hi,
        'Missing_Features': int(X.isna().sum().sum()),
        'Verdict': verdict,
        'Status': status,
        'Features_DF': X
    }
    return results, None

# ----------------------------------------------------------------------
# Application UI Layout
# ----------------------------------------------------------------------
st.title("📊 Nifty 500 Relative Valuation Dashboard")
st.markdown("Machine Learning-based peer valuation model trained on fundamental metrics.")

if saved_artifacts is None:
    st.error("Model file `nifty_valuation_model.pkl` not found. Please run `train_model.ipynb` first.")
    st.stop()

# Sidebar Setup
st.sidebar.header("Navigation & Options")
mode = st.sidebar.radio("Select View Mode", ["Single Stock Valuation", "Batch Screener"])

# ----------------------------------------------------------------------
# Mode 1: Single Stock Valuation
# ----------------------------------------------------------------------
if mode == "Single Stock Valuation":
    st.subheader("Single Ticker Analysis")
    ticker_input = st.text_input("Enter NSE Ticker Symbol (e.g. TCS.NS, INFY.NS, RELIANCE.NS):", value="TCS.NS").strip().upper()

    if st.button("Evaluate Stock", type="primary"):
        with st.spinner(f"Fetching financial data and evaluating {ticker_input}..."):
            data, error = evaluate_single_stock(ticker_input, saved_artifacts)

        if error:
            st.error(f"Failed to evaluate {ticker_input}: {error}")
        else:
            st.header(f"{data['Company_Name']} ({ticker_input})")
            st.caption(f"Sector: **{data['Sector']}**")

            # Status Banner
            if data['Status'] == "success":
                st.success(f"**Verdict:** {data['Verdict']}")
            elif data['Status'] == "warning":
                st.warning(f"**Verdict:** {data['Verdict']}")
            else:
                st.info(f"**Verdict:** {data['Verdict']}")

            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{data['Current_Price']:,.2f}")
            col2.metric("Actual Trailing P/E", f"{data['Actual_PE']:.2f}")
            col3.metric("Fair P/E (Model)", f"{data['Fair_PE']:.2f}")
            col4.metric("PE / Fair Ratio", f"{data['Actual_PE'] / data['Fair_PE']:.2f}x")

            st.divider()

            # Price & Valuation Ranges
            st.subheader("Valuation Range Summary")
            r_col1, r_col2 = st.columns(2)

            with r_col1:
                st.markdown("##### P/E Multiple Range (80% Confidence)")
                st.write(f"- **Minimum P/E:** `{data['PE_Min']:.2f}`")
                st.write(f"- **Fair P/E:** `{data['Fair_PE']:.2f}`")
                st.write(f"- **Maximum P/E:** `{data['PE_Max']:.2f}`")

            with r_col2:
                st.markdown("##### Estimated Fair Price Range")
                st.write(f"- **Lower Price Bound:** `₹{data['Price_Min']:,.2f}`")
                st.write(f"- **Fair Target Price:** `₹{data['Fair_Price']:,.2f}`")
                st.write(f"- **Upper Price Bound:** `₹{data['Price_Max']:,.2f}`")

            # Expandable Raw Input Data
            with st.expander("View Model Input Features"):
                st.dataframe(data['Features_DF'])

# ----------------------------------------------------------------------
# Mode 2: Batch Screener
# ----------------------------------------------------------------------
else:
    st.subheader("Batch Stock Screener")
    
    default_list = "TCS.NS, INFY.NS, HDFCBANK.NS, SBIN.NS, RELIANCE.NS, HINDUNILVR.NS, ITC.NS, SUNPHARMA.NS, LT.NS, MARUTI.NS"
    raw_tickers = st.text_area("Enter comma-separated tickers:", value=default_list, height=100)

    if st.button("Run Screener", type="primary"):
        ticker_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_list = []
        for i, t in enumerate(ticker_list):
            status_text.text(f"Evaluating {t} ({i+1}/{len(ticker_list)})...")
            data, err = evaluate_single_stock(t, saved_artifacts)
            if data:
                results_list.append({
                    'Ticker': t,
                    'Company': data['Company_Name'],
                    'Sector': data['Sector'],
                    'Price (₹)': round(data['Current_Price'], 2),
                    'Actual P/E': round(data['Actual_PE'], 2),
                    'Fair P/E': round(data['Fair_PE'], 2),
                    'P/E Min': round(data['PE_Min'], 2),
                    'P/E Max': round(data['PE_Max'], 2),
                    'P/E / Fair Ratio': round(data['Actual_PE'] / data['Fair_PE'], 2),
                    'Verdict': data['Verdict']
                })
            progress_bar.progress((i + 1) / len(ticker_list))
        
        status_text.success("Screening completed!")
        
        if results_list:
            df_res = pd.DataFrame(results_list)
            st.dataframe(df_res.sort_values("P/E / Fair Ratio"), use_container_width=True)