import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime
import pytz
import streamlit as st

warnings.filterwarnings("ignore")

ASSETS = {
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'SOL-USD': 'Solana',
    'EURUSD=X': 'EUR/USD', 'GBPUSD=X': 'GBP/USD', 'USDJPY=X': 'USD/JPY',
    'AUDUSD=X': 'AUD/USD', 'USDCAD=X': 'USD/CAD', 'USDCHF=X': 'USD/CHF',
    'NQ=F': 'Nasdaq 100', 'YM=F': 'US30 (Dow Jones)', '^GSPC': 'S&P 500',
    'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil', '^GDAXI': 'DAX 40'
}

def get_morocco_time():
    tz = pytz.timezone('Africa/Casablanca')
    return datetime.now(tz)

def get_relative_strength():
    check_pairs = {'EURUSD=X': 'EUR', 'GBPUSD=X': 'GBP', 'USDJPY=X': 'JPY', 'AUDUSD=X': 'AUD'}
    s_map = {'USD': 0.0}
    for sym, name in check_pairs.items():
        try:
            d = yf.download(sym, period='5d', interval='15m', progress=False)
            if d.empty: continue
            change = ((d['Close'].iloc[-1] - d['Close'].iloc[-20]) / d['Close'].iloc[-20]) * 100
            s_map[name] = -float(change) if name == 'JPY' else float(change)
        except: 
            s_map[name] = 0
    return s_map

def compute_engine(df, symbol, s_map):
    df = df.copy()
    df['ATR'] = pd.concat([
        df['High']-df['Low'], 
        (df['High']-df['Close'].shift()).abs(), 
        (df['Low']-df['Close'].shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Returns'] = df['Close'].pct_change(1)
    df['CS'] = 0.0
    for curr, val in s_map.items():
        if curr in symbol: 
            df['CS'] = val if symbol.startswith(curr) else -val

    df['Target'] = np.select([
        (df['Close'].shift(-4)-df['Close'] > df['ATR']*0.5), 
        (df['Close'].shift(-4)-df['Close'] < -df['ATR']*0.5)
    ], [1, 2], default=0)
    
    return df.dropna()

def get_ai_prediction(df):
    features = ['RSI', 'Returns', 'CS']
    X = df[features].values
    y = df['Target'].values
    
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, verbosity=0, random_state=42)
    model.fit(X[:-4], y[:-4])
    
    probs = model.predict_proba(X[-1:])[0]
    prediction = np.argmax(probs)
    confidence = probs[prediction] * 100
    return prediction, confidence

def run_analysis():
    s_map = get_relative_strength()
    results = []

    for sym, name in ASSETS.items():
        try:
            d15_raw = yf.download(sym, period='40d', interval='15m', progress=False)
            if d15_raw.empty: continue
            df = compute_engine(d15_raw, sym, s_map)
            pred, conf = get_ai_prediction(df)
            
            current_p = float(df['Close'].iloc[-1])
            sig = "BUY 🚀" if pred==1 else "SELL 📉"
            
            results.append({
                'ASSET': name,
                'PRICE': f"{current_p:.2f}",
                'SIGNAL': sig,
                'CONF%': f"{conf:.1f}%"
            })
        except:
            continue
    return results

# ==================== STREAMLIT ====================
st.set_page_config(page_title="SIDRA QUANT v14", layout="wide")

st.title("SIDRA QUANT v14 - تحليلات الأسواق المالية")

if st.button("🚀 بدء التحليل"):
    with st.spinner("جاري تحليل الأسواق..."):
        results = run_analysis()
    
    if results:
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("❌ لم يتم العثور على بيانات. تحقق من اتصال الإنترنت.")
else:
    st.info("👆 اضغط على زر 'بدء التحليل' لتحليل الأسواق")