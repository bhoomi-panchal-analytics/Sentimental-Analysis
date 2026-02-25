# app.py

import streamlit as st
import torch
import os

from config import RSS_FEEDS, VIX_TICKER, LOOKBACK_PERIOD, INTERVAL
from cnn_model import ChartCNN
from lstm_model import LSTMForecaster
from chart_to_image import price_to_image
from sentiment import aggregate_sentiment
from ss_news import fetch_rss   # IMPORTANT: matches your GitHub file name
from market_data import get_market_data
from inference_engine import run_lstm_prediction

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Volatility Engine", layout="wide")
st.title("Real-Time Sentiment Augmented Volatility Monitor")

# --------------------------------------------------
# MODEL LOADING (SAFE)
# --------------------------------------------------

@st.cache_resource
def load_models():

    cnn = ChartCNN()
    lstm = LSTMForecaster(input_size=5)

    # Load weights only if they exist
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


cnn_model, lstm_model = load_models()

# --------------------------------------------------
# DATA FETCHING (CACHED)
# --------------------------------------------------

@st.cache_data(ttl=60)
def get_news():
    headlines = []
    for feed in RSS_FEEDS:
        try:
            headlines.extend(fetch_rss(feed))
        except:
            pass
    return headlines


@st.cache_data(ttl=60)
def get_market():
    try:
        return get_market_data(
            ticker=VIX_TICKER,
            period=LOOKBACK_PERIOD,
            interval=INTERVAL
        )
    except:
        return None


news_headlines = get_news()
market_df = get_market()

if market_df is None or len(market_df) < 30:
    st.error("Market data unavailable.")
    st.stop()

# --------------------------------------------------
# SENTIMENT
# --------------------------------------------------

sentiment_score = 0.0

if news_headlines:
    try:
        sentiment_score = aggregate_sentiment(news_headlines)
    except:
        sentiment_score = 0.0

# --------------------------------------------------
# CNN ANALYSIS
# --------------------------------------------------

cnn_prob = 0.0

try:
    image_tensor = price_to_image(market_df)

    with torch.no_grad():
        cnn_prob = cnn_model(image_tensor).item()

except Exception as e:
    st.warning(f"CNN inference skipped: {e}")

# --------------------------------------------------
# LSTM FORECAST
# --------------------------------------------------

lstm_prediction = 0.0

try:
    pred = run_lstm_prediction(market_df, lstm_model)
    if pred is not None:
        lstm_prediction = float(pred)
except:
    pass

# --------------------------------------------------
# COMPOSITE SIGNAL
# --------------------------------------------------

composite_signal = (
    0.4 * sentiment_score +
    0.3 * cnn_prob +
    0.3 * lstm_prediction
)

# --------------------------------------------------
# DASHBOARD DISPLAY
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Sentiment Score", round(sentiment_score, 4))
col2.metric("CNN Pattern Probability", round(cnn_prob, 4))
col3.metric("LSTM Forecast", round(lstm_prediction, 4))
col4.metric("Composite Signal", round(composite_signal, 4))

st.subheader("Live VIX Price")
st.line_chart(market_df["Close"])

# --------------------------------------------------
# RISK INTERPRETATION
# --------------------------------------------------

if composite_signal > 0.4:
    st.error("High Volatility Regime")
elif composite_signal < -0.3:
    st.success("Low Volatility Regime")
else:
    st.warning("Neutral Volatility Zone")

# --------------------------------------------------
# HEADLINES DISPLAY
# --------------------------------------------------

st.subheader("Headlines Used")

for h in news_headlines[:10]:
    st.write("•", h)
