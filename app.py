# app.py

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

from config import RSS_FEEDS, VIX_TICKER, LOOKBACK_PERIOD, INTERVAL
from cnn_model import ChartCNN
from lstm_model import LSTMForecaster
from chart_to_image import price_to_image
from sentiment import aggregate_sentiment
from ss_news import fetch_rss
from market_data import get_market_data
from inference_engine import run_lstm_prediction
from feature_engineering import create_features

st.set_page_config(layout="wide")
st.title("Sentiment-Augmented Volatility Intelligence System")

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    cnn = ChartCNN()
    lstm = LSTMForecaster(input_size=5)

    if os.path.exists("models/cnn_weights.pth"):
        cnn.load_state_dict(torch.load("models/cnn_weights.pth", map_location="cpu"))

    if os.path.exists("models/lstm_weights.pth"):
        lstm.load_state_dict(torch.load("models/lstm_weights.pth", map_location="cpu"))

    cnn.eval()
    lstm.eval()

    return cnn, lstm

cnn_model, lstm_model = load_models()

# ---------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_news():
    headlines = []
    for feed in RSS_FEEDS:
        try:
            headlines.extend(fetch_rss(feed))
        except:
            pass
    return headlines

@st.cache_data(ttl=60)
def fetch_market():
    return get_market_data(
        ticker=VIX_TICKER,
        period="6mo",
        interval="1d"
    )

news_headlines = fetch_news()
market_df = fetch_market()

if market_df is None or len(market_df) < 50:
    st.stop()

# ---------------------------------------------------------
# SENTIMENT DECOMPOSITION
# ---------------------------------------------------------

positive, negative, neutral = 0, 0, 0
scores = []

for h in news_headlines[:20]:
    score = aggregate_sentiment([h])
    scores.append(score)
    if score > 0:
        positive += 1
    elif score < 0:
        negative += 1
    else:
        neutral += 1

sentiment_score = np.mean(scores) if scores else 0

# ---------------------------------------------------------
# CNN + LSTM
# ---------------------------------------------------------

image_tensor = price_to_image(market_df)
cnn_prob = cnn_model(image_tensor).item()

lstm_prediction = run_lstm_prediction(market_df, lstm_model)
if lstm_prediction is None:
    lstm_prediction = 0

# Confidence interval
forecast_std = market_df["Close"].pct_change().rolling(20).std().iloc[-1]
upper_band = lstm_prediction + forecast_std
lower_band = lstm_prediction - forecast_std

composite_signal = 0.4*sentiment_score + 0.3*cnn_prob + 0.3*lstm_prediction

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Live Monitor", "Research Panel", "Strategy Lab", "Diagnostics"])

# =========================================================
# TAB 1 — LIVE MONITOR
# =========================================================

with tab1:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sentiment", round(sentiment_score,4))
    col2.metric("CNN Pattern Prob", round(cnn_prob,4))
    col3.metric("LSTM Forecast", round(lstm_prediction,4))
    col4.metric("Composite Signal", round(composite_signal,4))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=market_df.index, y=market_df["Close"], name="VIX"))
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment stacked bar
    sentiment_df = pd.DataFrame({
        "Type":["Positive","Negative","Neutral"],
        "Count":[positive, negative, neutral]
    })
    st.subheader("Sentiment Breakdown")
    st.bar_chart(sentiment_df.set_index("Type"))

# =========================================================
# TAB 2 — RESEARCH PANEL
# =========================================================

with tab2:

    market_df = create_features(market_df)

    # Rolling correlation
    market_df["sentiment_proxy"] = sentiment_score
    market_df["rolling_corr"] = market_df["Close"].pct_change().rolling(20).corr(
        market_df["sentiment_proxy"]
    )

    st.subheader("Rolling 20D Correlation: Sentiment vs VIX")
    st.line_chart(market_df["rolling_corr"])

    # Volatility Regime Classification
    vol = market_df["Close"]
    regime = pd.qcut(vol, q=3, labels=["Low","Medium","High"])
    regime_df = pd.DataFrame({"VIX":vol, "Regime":regime})

    fig = px.scatter(regime_df, x=regime_df.index, y="VIX", color="Regime")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 — STRATEGY LAB
# =========================================================

with tab3:

    threshold = st.slider("Signal Threshold", 0.0, 1.0, 0.3)

    signals = np.where(composite_signal > threshold, -1, 1)
    returns = market_df["Close"].pct_change().fillna(0)

    strategy_returns = signals * returns
    equity_curve = (1 + strategy_returns).cumprod()

    st.line_chart(equity_curve)

    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    st.metric("Strategy Sharpe Ratio", round(sharpe,3))

# =========================================================
# TAB 4 — DIAGNOSTICS
# =========================================================

with tab4:

    st.write("Headlines Used:", len(news_headlines))
    st.write("Data Points:", len(market_df))

    st.subheader("Top 3 Most Negative Headlines")

    ranked = sorted(zip(news_headlines, scores), key=lambda x: x[1])
    for h, s in ranked[:3]:
        st.write(f"{round(s,3)} → {h}")

    # Download signals
    export_df = pd.DataFrame({
        "Date": market_df.index,
        "VIX": market_df["Close"],
        "CompositeSignal": composite_signal
    })

    st.download_button(
        label="Download Signals CSV",
        data=export_df.to_csv(index=False),
        file_name="volatility_signals.csv",
        mime="text/csv"
    )
