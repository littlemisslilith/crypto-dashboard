import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD

# --- Sidebar Inputs ---
st.sidebar.title("Crypto Forecast Settings")
symbol = st.sidebar.selectbox("Choose Symbol", ["ETH-USD", "BTC-USD", "SOL-USD"])
mu = st.sidebar.slider("Expected Return (μ)", 0.00, 0.02, 0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.00, 0.10, 0.06)
phi = st.sidebar.slider("Phi (Drift Adj)", 0.90, 1.10, 1.00)
lambda_ = st.sidebar.slider("Lambda (Rate)", 0.10, 2.00, 1.00)

# --- Load Data ---
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="30d", interval="1h")
    df.dropna(inplace=True)
    return df

df = load_data(symbol)

# --- Handle empty data ---
if df.empty or "Close" not in df.columns:
    st.error("No data loaded. Try a different symbol or check your connection.")
    st.stop()

# --- Technical Indicators ---
df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
macd = MACD(close=df["Close"])
df["MACD"] = macd.macd()
df["Signal"] = macd.macd_signal()

# --- Forecasting with Enhanced GBM ---
S0 = df["Close"].iloc[-1]
T = 24 / 24  # 24h forecast in 1-day
dt = 1 / 24
phi_lambda = phi * lambda_
Z = np.random.normal(size=24)
t_values = np.arange(1, 25) * dt
S = S0 * np.exp((mu - 0.5 * sigma**2) * t_values + sigma * np.sqrt(t_values) * phi_lambda * Z)

# --- Layout ---
st.title("🔮 Crypto Forecast Dashboard")
st.subheader(f"Live Forecast for {symbol}")

# --- Live Price Chart ---
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
fig1.add_tr_
