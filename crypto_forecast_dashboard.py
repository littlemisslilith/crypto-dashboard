
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
from datetime import datetime
from io import BytesIO

# --- SETTINGS ---
ETH_SYMBOL = "ethereum"
BTC_SYMBOL = "bitcoin"
VS_CURRENCY = "usd"

# --- FUNCTIONS ---
def get_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies={VS_CURRENCY}"
    res = requests.get(url)
    return res.json()[symbol][VS_CURRENCY]

def gbm_forecast(S0, mu, sigma, days=30, n_sim=50):
    dt = 1
    prices = np.zeros((days, n_sim))
    for sim in range(n_sim):
        S = [S0]
        for _ in range(days - 1):
            drift = (mu - 0.5 * sigma**2) * dt
            shock = sigma * np.random.normal() * np.sqrt(dt)
            S.append(S[-1] * np.exp(drift + shock))
        prices[:, sim] = S
    return prices

def generate_commentary(price_now, mu, sigma, coin="ETH"):
    direction = "increase 📈" if mu > 0 else "decrease 📉" if mu < 0 else "stay relatively stable"
    volatility = "high volatility ⚠️" if sigma > 0.05 else "low volatility ✅"
    return f"{coin} is expected to {direction} over the next 30 days with {volatility}. Current price is ${price_now:.2f}."

# --- STREAMLIT DASHBOARD ---
st.set_page_config(page_title="Crypto Forecast Dashboard", layout="centered")
st.title("📊 Crypto Forecast Dashboard")

# Get current prices
eth_price = get_price(ETH_SYMBOL)
btc_price = get_price(BTC_SYMBOL)

# Show current prices
col1, col2 = st.columns(2)
with col1:
    st.subheader("💰 ETH Price")
    st.metric("Current ETH (USD)", f"${eth_price:,.2f}")
with col2:
    st.subheader("💰 BTC Price")
    st.metric("Current BTC (USD)", f"${btc_price:,.2f}")

st.markdown("---")
st.header("📈 Monte Carlo Forecast Inputs")
mu = st.number_input("Expected Daily Drift (μ)", value=0.002, step=0.0001)
sigma = st.number_input("Expected Daily Volatility (σ)", value=0.04, step=0.001)

# Simulate prices
eth_forecast = gbm_forecast(eth_price, mu, sigma)
btc_forecast = gbm_forecast(btc_price, mu, sigma)

# Plot ETH & BTC forecasts
st.subheader("📉 30-Day Forecasts")
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(eth_forecast.shape[1]):
    ax.plot(eth_forecast[:, i], color='purple', alpha=0.3, linewidth=1)
for i in range(btc_forecast.shape[1]):
    ax.plot(btc_forecast[:, i], color='orange', alpha=0.3, linewidth=1)
ax.set_title("Simulated Price Paths for ETH (purple) and BTC (orange)")
ax.set_ylabel("Price (USD)")
st.pyplot(fig)

# Generate & display AI commentary
st.markdown("---")
st.subheader("🧠 AI Commentary")
st.write(generate_commentary(eth_price, mu, sigma, "ETH"))
st.write(generate_commentary(btc_price, mu, sigma, "BTC"))

# Export to CSV
if st.button("📤 Export Forecast to CSV"):
    df = pd.DataFrame({
        f"ETH_sim_{i+1}": eth_forecast[:, i] for i in range(eth_forecast.shape[1])
    } | {
        f"BTC_sim_{i+1}": btc_forecast[:, i] for i in range(btc_forecast.shape[1])
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "forecast.csv", "text/csv")

# Timestamp
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
