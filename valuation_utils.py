import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

# Feature definitions
NUM_FEATURES = [
    'ROE', 'ROA', 'Operating_Margin', 'Net_Profit_Margin',
    'Revenue_Growth', 'Earnings_Growth', 'Debt_To_Equity',
    'Current_Ratio', 'FCF_Margin', 'Log_Revenue',
    'Revenue_CAGR', 'Earnings_CAGR', 'Earnings_CV',
    'Beta', 'Payout_Ratio'
]
CAT_FEATURES = ['Sector']
FEATURES = NUM_FEATURES + CAT_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Consistent feature engineering applied during both training and inference."""
    out = df.copy()

    non_numeric = ['Ticker', 'Company_Name', 'Sector', 'Industry']
    num_cols = [c for c in out.columns if c not in non_numeric]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors='coerce')

    out = out.replace([np.inf, -np.inf], np.nan)
    out['Sector'] = out['Sector'].fillna('Unknown')

    rev = out['Total_Revenue'].where(out['Total_Revenue'] > 0)
    out['FCF_Margin'] = out['Free_Cash_Flow'] / rev
    out['Log_Revenue'] = np.log(rev)

    financial_mask = out['Sector'].eq('Financial Services')
    out.loc[financial_mask, ['Debt_To_Equity', 'Current_Ratio']] = np.nan
    return out


class Winsorizer(BaseEstimator, TransformerMixin):
    """Calculates quantiles exclusively on training folds to prevent data leakage (NaN-safe)."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X_arr, self.lower, axis=0)
        self.hi_ = np.nanquantile(X_arr, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class SectorMedianRegressor(BaseEstimator, RegressorMixin):
    """Industry benchmark baseline: predicts using sector P/E median."""
    def fit(self, X, y):
        self.medians_ = pd.Series(np.asarray(y), index=X['Sector'].values).groupby(level=0).median()
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        return X['Sector'].map(self.medians_).fillna(self.global_).to_numpy()


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _calculate_cagr(series):
    series = series.dropna().sort_index()
    if len(series) < 3:
        return np.nan
    start, end, n = series.iloc[0], series.iloc[-1], len(series) - 1
    if start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / n) - 1


def _extract_row(financials, names):
    for name in names:
        if name in financials.index:
            return pd.to_numeric(financials.loc[name], errors='coerce')
    return pd.Series(dtype=float)


def fetch_growth_features(ticker_symbol, info=None) -> dict:
    """Extracts risk and historical growth features from yfinance."""
    out = {
        'Ticker': ticker_symbol, 'Revenue_CAGR': np.nan, 'Earnings_CAGR': np.nan,
        'Earnings_CV': np.nan, 'Beta': np.nan, 'Payout_Ratio': np.nan
    }
    try:
        tk = yf.Ticker(ticker_symbol)
        info = info if info is not None else tk.info
        out['Beta'] = _safe_float(info.get('beta'))
        out['Payout_Ratio'] = _safe_float(info.get('payoutRatio'))

        fin = tk.financials
        if fin is not None and not fin.empty:
            rev = _extract_row(fin, ['Total Revenue', 'Operating Revenue'])
            ni = _extract_row(fin, ['Net Income', 'Net Income Common Stockholders'])
            out['Revenue_CAGR'] = _calculate_cagr(rev)
            out['Earnings_CAGR'] = _calculate_cagr(ni)
            ni_clean = ni.dropna()
            if len(ni_clean) >= 3 and ni_clean.abs().mean() > 0:
                out['Earnings_CV'] = ni_clean.std() / ni_clean.abs().mean()
    except Exception:
        pass
    return out