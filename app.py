import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set page config for wider layout and page title
st.set_page_config(layout="wide", page_title="Advanced Options Pricing")

# Define custom colors
primary_color = "#E63946"
secondary_color = "#F1FAEE"
background_color = "#A8DADC"
text_color = "#1D3557"

# Custom CSS to inject into the Streamlit app
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
    }}
    h1 {{
        color: {primary_color};
    }}
    .reportview-container .sidebar .sidebar-content {{
        background-color: {secondary_color};
    }}
    .css-1aumxhk {{
        background-color: {secondary_color};
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)


# Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# Binomial Model
def binomial_option(S, K, T, r, sigma, steps, option_type='call'):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    price_tree = np.zeros([steps + 1, steps + 1])

    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = S * (u ** j) * (d ** (i - j))

    option_tree = np.zeros([steps + 1, steps + 1])
    if option_type == 'call':
        option_tree[:, steps] = np.maximum(np.zeros(steps + 1), price_tree[:, steps] - K)
    else:
        option_tree[:, steps] = np.maximum(np.zeros(steps + 1), K - price_tree[:, steps])

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])

    return option_tree[0, 0]


# Function to calculate Greeks (placeholder implementation)
def calculate_greeks(S, K, T, r, sigma, option_type):
    # Actual Greek calculations should be implemented here
    # Placeholder values are returned for demonstration
    return {"Delta": 0.5, "Gamma": 0.1, "Theta": -0.01, "Vega": 0.2, "Rho": 0.15}


# Streamlit UI
def main():
    st.title("🌟 Advanced Option Pricing App 🌟")

    # Sidebar for Input Parameters
    with st.sidebar:
        st.header("📊 Option Parameters")
        S = st.number_input("Stock Price", value=100.0, step=0.01)
        K = st.number_input("Strike Price", value=100.0, step=0.01)
        T = st.number_input("Time to Maturity (in months)", value=0.25, step=0.01)
        T = T / 12  # Convert months to years
        r = st.number_input("Risk-free Rate", value=0.05, step=0.01)
        sigma = st.number_input("Volatility", value=0.2, step=0.01)
        option_type = st.selectbox("Option Type", ['call', 'put'])

    # Calculate option prices
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    binomial_price = binomial_option(S, K, T, r, sigma, steps=100, option_type=option_type)
    greeks = calculate_greeks(S, K, T, r, sigma, option_type)

    # Displaying the results
    st.write("### Option Prices")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Black-Scholes Price", value=f"{bs_price:.2f}")
    with col2:
        st.metric(label="Binomial Model Price", value=f"{binomial_price:.2f}")

    # Greeks
    st.write("### Greeks")
    for key, value in greeks.items():
        st.write(f"{key}: {value:.2f}")

    # Display Historical Stock Data
    st.subheader("📈 Historical Stock Data")
    stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    start_date = datetime.now() - timedelta(days=2 * 365)
    end_date = datetime.now()

    for symbol in stock_symbols:
        with st.expander(f"{symbol} Stock Data"):
            data = yf.download(symbol, start=start_date, end=end_date)
            st.line_chart(data['Close'])


if __name__ == "__main__":
    main()

# Author: Francesco Tassia
