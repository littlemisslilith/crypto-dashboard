import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
from ta.momentum import RSIIndicator

# ----------------- SETTINGS ------------------
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# -------------- HEADER -----------------------
st.title("🧠 Crypto Forecast Dashboard")
st.caption("by Tia & GPT, powered by Streamlit + Alpha sauce")

# -------- USER INPUT SECTION -----------------
symbol = st.selectbox("Choose Symbol", ["ETH/USDT", "BTC/USDT"])
mu = st.slider("Expected Return (μ)", 0.0, 0.02, 0.01)
sigma = st.slider("Volatility (σ)", 0.0, 0.1, 0.06)
phi = st.slider("Phi (Drift Adj)", 0.90, 1.10, 1.00)
_lambda = st.slider("Lambda (Rate)", 0.1, 2.0, 1.0)

# -------- FETCH PRICE DATA -------------------
yf_symbol = symbol.replace("/", "-")  # e.g., ETH/USDT → ETH-USDT
df = yf.download(yf_symbol, period="7d", interval="1h")

# -------- HANDLE MISSING or EMPTY DATA -------
if df.empty or "Close" not in df.columns:
    st.error("⚠️ No valid data fetched. Check symbol or try again later.")
    st.stop()

# Normalize column names for safety
df.columns = [col.capitalize() for col in df.columns]

# -------- CALCULATE RSI ----------------------
rsi = RSIIndicator(close=df["Close"])
df["RSI"] = rsi.rsi()

# -------- PRICE FORECAST USING GBM ----------
S0 = df["Close"].iloc[-1]
T = 1  # forecast 1 day
N = 24  # hourly steps
dt = T / N

forecast = []
price = S0

for _ in range(N):
    shock = np.random.normal()
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * shock * phi * _lambda
    price *= np.exp(drift + diffusion)
    forecast.append(price)

# -------- PLOT PRICE + FORECAST --------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Historical'))
fig.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=N+1, freq='H')[1:], y=forecast, mode='lines', name='Forecast'))

fig.update_layout(title=f"{symbol} Forecast", xaxis_title="Time", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# -------- RSI GRAPH --------------------------
st.subheader("📈 RSI Indicator")
st.line_chart(df["RSI"])
