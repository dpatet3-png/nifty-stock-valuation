import pickle
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Configurazione della pagina web
st.set_page_config(page_title="Nifty Stock Evaluator", page_icon="📈", layout="centered")

st.title("📈 Nifty 500 Stock Valuation Engine")
st.markdown("Valutazione statistica in tempo reale basata su **Peer Multiples** e **Random Forest**.")

# Caricamento del modello salvato (.pkl)
@st.cache_resource
def load_model():
    with open("nifty_valuation_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    saved_data = load_model()
    model = saved_data['model']
    feature_cols = saved_data['features']
    st.sidebar.success("✅ Modello ML caricato correttamente!")
except Exception as e:
    st.error(f"Errore nel caricamento del file .pkl: {e}")
    st.stop()

# Casella di input per l'utente
ticker_input = st.text_input("Inserisci il Ticker azionario (es. RELIANCE, INFY, TCS):", "TCS").strip().upper()

if st.button("Analizza Azione", type="primary"):
    ticker_symbol = ticker_input if ticker_input.endswith(".NS") else f"{ticker_input}.NS"
    
    with st.spinner(f"Download dati e calcolo Fair Value per {ticker_symbol}..."):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
                st.error(f"⚠️ Impossibile trovare dati validi per **{ticker_symbol}**.")
            else:
                current_price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
                trailing_pe = info.get('trailingPE', np.nan)
                
                data = {
                    'ROE': info.get('returnOnEquity', np.nan),
                    'ROA': info.get('returnOnAssets', np.nan),
                    'Operating_Margin': info.get('operatingMargins', np.nan),
                    'Net_Profit_Margin': info.get('profitMargins', np.nan),
                    'Revenue_Growth': info.get('revenueGrowth', np.nan),
                    'Earnings_Growth': info.get('earningsGrowth', np.nan),
                    'Debt_To_Equity': info.get('debtToEquity', np.nan),
                    'Current_Ratio': info.get('currentRatio', np.nan)
                }
                
                df_input = pd.DataFrame([data]).fillna(0)
                predicted_fair_pe = model.predict(df_input[feature_cols])[0]
                
                if pd.isna(trailing_pe) or trailing_pe <= 0:
                    st.warning(f"Trailing P/E non disponibile o negativo. Fair P/E stimato: **{predicted_fair_pe:.2f}**")
                else:
                    fair_price = current_price * (predicted_fair_pe / trailing_pe)
                    diff_percent = ((current_price - fair_price) / fair_price) * 100
                    
                    st.divider()
                    st.subheader(f"🏢 {info.get('shortName', ticker_symbol)}")
                    st.caption(f"Settore: {info.get('sector', 'N/A')} | Industria: {info.get('industry', 'N/A')}")
                    
                    # Visualizzazione delle metriche principali in colonne
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prezzo Reale", f"₹{current_price:.2f}")
                    col2.metric("Fair Value Stimato", f"₹{fair_price:.2f}")
                    col3.metric("Scostamento", f"{diff_percent:+.2f}%")
                    
                    # Verdetto visivo
                    st.subheader("Verdetto Statistico:")
                    if diff_percent < -15:
                        st.success("🟢 **SOTTOVALUTATA** — L'azione scambia a sconto rispetto ai fondamentali dei peer.")
                    elif diff_percent > 15:
                        st.error("🔴 **SOPRAVVALUTATA** — L'azione scambia a premio elevato rispetto ai peer.")
                    else:
                        st.info("🟡 **CORRETTAMENTE VALUTATA** — Prezzo in linea con i multipli di settore.")
                    
                    # Dettagli dei multipli in un menu a tendina
                    with st.expander("Vedi dettagli multipli di mercato"):
                        st.write(f"- **Trailing P/E Reale**: {trailing_pe:.2f}")
                        st.write(f"- **Fair P/E Predetto (ML)**: {predicted_fair_pe:.2f}")
                        st.write(f"- **Range di Confidenza (±15%)**: ₹{fair_price*0.85:.2f} — ₹{fair_price*1.15:.2f}")
        except Exception as e:
            st.error(f"Errore durante l'elaborazione: {e}")