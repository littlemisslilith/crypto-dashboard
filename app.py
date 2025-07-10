import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands
import math

# --- CONFIG ---
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
symbols = {"ETH/USDT": "ETH-USD", "BTC/USDT": "BTC-USD", "SOL/USDT": "SOL-USD"}
symbol = st.sidebar.selectbox("Choose Symbol", list(symbols.keys()))
symbol_yf = symbols[symbol]

mu = st.sidebar.slider("Expected Return (μ)", 0.0000, 0.0200, 0.0100)
sigma = st.sidebar.slider("Volatility (σ)", 0.000, 0.100, 0.060)
phi = st.sidebar.slider("Phi (φ)", 0.90, 1.10, 1.00)
lambda_ = st.sidebar.slider("Lambda (λ)", 0.1, 2.0, 1.0)

# --- GET DATA ---
@st.cache_data(ttl=60)
def get_data(ticker):
    df = yf.download(ticker, period="7d", interval="1h")
    df.dropna(inplace=True)
    return df

df = get_data(symbol_yf)
latest_price = df["Close"].iloc[-1]

# --- INDICATORS ---
df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
df["MACD_diff"] = MACD(close=df["Close"]).macd_diff()
df["EMA50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
df["EMA200"] = EMAIndicator(close=df["Close"], window=200).ema_indicator()
bb = BollingerBands(close=df["Close"])
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()
ichimoku = IchimokuIndicator(high=df["High"], low=df["Low"])
df["cloud_lead1"] = ichimoku.ichimoku_a()
df["cloud_lead2"] = ichimoku.ichimoku_b()

# --- PREMIUM/DISCOUNT ---
premium = df["Close"].rolling(24).max()
discount = df["Close"].rolling(24).min()

# --- LAYOUT ---
st.title(f"{symbol} Smart Dashboard")
col1, col2 = st.columns([3, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=premium, name="Premium", line=dict(dash="dot", color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=discount, name="Discount", line=dict(dash="dot", color="red")))
    fig.update_layout(height=500, title="Price & Zones")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Market Info")
    st.metric("Last Price", f"${latest_price:,.2f}")
    st.metric("24H Change", f"{(df['Close'].pct_change().iloc[-1] * 100):.2f}%")
    st.write("RSI:", round(df["RSI"].iloc[-1], 2))
    st.write("MACD:", round(df["MACD_diff"].iloc[-1], 5))
    st.write("EMA Cross:", "Bullish" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "Bearish")

# --- ALERT PANEL ---
st.subheader("📢 Indicator Alerts")
if df["RSI"].iloc[-1] > 70:
    st.error("RSI: Overbought")
elif df["RSI"].iloc[-1] < 30:
    st.success("RSI: Oversold")
else:
    st.info("RSI: Neutral")
st.write("MACD:", "Bullish" if df["MACD_diff"].iloc[-1] > 0 else "Bearish")

# --- GBM FORECAST ---
st.subheader("🔮 GBM Forecast (24H)")
T = 1  # 1 day
S0 = latest_price
forecast = S0 * math.exp((mu - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * phi * lambda_)
var = sigma**2 * T
sd = forecast * (math.sqrt(math.exp(var) - 1))

low = round(forecast - sd, 2)
high = round(forecast + sd, 2)
st.write(f"**Expected price in 24H:** `${forecast:,.2f}`")
st.write(f"**Confidence range (68%):** `${low}` to `${high}`")

# --- PER-HOUR FORECAST ---
st.subheader("⏱️ GBM Per-Hour Forecast")
hours = np.arange(1, 25)
per_hour = [S0 * math.exp((mu - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * phi * lambda_) for t in hours]
hourly_df = pd.DataFrame({"Hour": hours, "Forecast": per_hour})

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hourly_df["Hour"], y=hourly_df["Forecast"], name="1H Forecast", line=dict(color="purple")))
fig2.update_layout(title="Hourly Forecast", xaxis_title="Hour", yaxis_title="Price", height=400)
st.plotly_chart(fig2, use_container_width=True)

# --- BACKTESTER ---
st.subheader("📈 Backtest Signals")
df["Buy"] = (df["RSI"] < 30) & (df["MACD_diff"] > 0)
df["Sell"] = (df["RSI"] > 70) & (df["MACD_diff"] < 0)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
fig3.add_trace(go.Scatter(x=df[df["Buy"]].index, y=df[df["Buy"]]["Close"], mode="markers", name="Buy", marker=dict(color="green", size=10)))
fig3.add_trace(go.Scatter(x=df[df["Sell"]].index, y=df[df["Sell"]]["Close"], mode="markers", name="Sell", marker=dict(color="red", size=10)))
fig3.update_layout(title="Backtest Strategy", height=500)
st.plotly_chart(fig3, use_container_width=True)
