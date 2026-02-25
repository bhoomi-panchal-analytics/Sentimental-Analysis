# app.py

import streamlit as st
import torch
import numpy as np
import pandas as pd

from config import RSS_FEEDS, VIX_TICKER, LOOKBACK_PERIOD, INTERVAL
from cnn_model import ChartCNN
from chart_to_image import price_to_image
from sentiment import aggregate_sentiment
from rss_news import fetch_rss
from market_data import get_market_data
from lstm_model import load_lstm
from inference_engine import run_lstm_prediction
from utils import safe_execution

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Real-Time Sentiment Volatility Engine",
    layout="wide"
)

st.title("Sentiment-Augmented Volatility Monitor")

# ---------------------------------------------------
# LOAD MODELS (CACHED)
# ---------------------------------------------------


import os

@st.cache_resource
def load_models():
    cnn = ChartCNN()
    lstm = LSTMForecaster(input_size=5)

    if os.path.exists("models/cnn_weights.pth"):
        cnn.load_state_dict(
            torch.load("models/cnn_weights.pth", map_location="cpu")
        )

    if os.path.exists("models/lstm_weights.pth"):
        lstm.load_state_dict(
            torch.load("models/lstm_weights.pth", map_location="cpu")
        )

    cnn.eval()
    lstm.eval()

    return cnn, lstm

# ---------------------------------------------------
# DATA FETCHING (CACHED 60s)
# ---------------------------------------------------

@st.cache_data(ttl=60)
def get_cached_news():
    headlines = []
    for feed in RSS_FEEDS:
        data = safe_execution(lambda: fetch_rss(feed))
        if data:
            headlines.extend(data)
    return headlines


@st.cache_data(ttl=60)
def get_cached_market():
    return safe_execution(
        lambda: get_market_data(
            ticker=VIX_TICKER,
            period=LOOKBACK_PERIOD,
            interval=INTERVAL
        )
    )

# ---------------------------------------------------
# FETCH DATA
# ---------------------------------------------------

news_headlines = get_cached_news()
market_df = get_cached_market()

if market_df is None or len(market_df) < 30:
    st.error("Market data unavailable.")
    st.stop()

# ---------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------

if news_headlines:
    sentiment_score = aggregate_sentiment(news_headlines)
else:
    sentiment_score = 0.0

# ---------------------------------------------------
# CNN CHART ANALYSIS
# ---------------------------------------------------

image_tensor = price_to_image(market_df)

with torch.no_grad():
    cnn_prob = cnn_model(image_tensor).item()

# ---------------------------------------------------
# LSTM FORECAST
# ---------------------------------------------------

lstm_prediction = run_lstm_prediction(market_df, lstm_model)

if lstm_prediction is None:
    lstm_prediction = 0.0

# ---------------------------------------------------
# COMPOSITE SIGNAL
# ---------------------------------------------------

composite_signal = (
    0.4 * sentiment_score +
    0.3 * cnn_prob +
    0.3 * lstm_prediction
)

# ---------------------------------------------------
# DASHBOARD DISPLAY
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Sentiment Score", round(sentiment_score, 4))
col2.metric("CNN Pattern Prob", round(cnn_prob, 4))
col3.metric("LSTM Forecast", round(lstm_prediction, 4))
col4.metric("Composite Signal", round(composite_signal, 4))

st.subheader("Live VIX Price")
st.line_chart(market_df["Close"])

# ---------------------------------------------------
# RISK INTERPRETATION
# ---------------------------------------------------

if composite_signal > 0.4:
    st.error("High Volatility Regime")
elif composite_signal < -0.3:
    st.success("Low Volatility Regime")
else:
    st.warning("Neutral Volatility Zone")

# ---------------------------------------------------
# SHOW LATEST HEADLINES
# ---------------------------------------------------

st.subheader("Latest Headlines Used in Sentiment")

for h in news_headlines[:10]:
    st.write("•", h)
