import streamlit as st
import torch
from cnn_model import ChartCNN
from chart_to_image import price_to_image
from sentiment import aggregate_sentiment
from rss_news import fetch_rss
from market_data import get_market_data

st.set_page_config(layout="wide")

st.title("Real-Time Sentiment-Augmented Volatility Monitor")

@st.cache_resource
def load_cnn():
    model = ChartCNN()
    model.load_state_dict(torch.load("cnn_weights.pth", map_location="cpu"))
    model.eval()
    return model

model = load_cnn()

rss_sources = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://finance.yahoo.com/rss/"
]

all_headlines = []
for src in rss_sources:
    all_headlines.extend(fetch_rss(src))

sentiment_score = aggregate_sentiment(all_headlines)

st.metric("Live Sentiment Score", round(sentiment_score, 3))

df = get_market_data("^VIX")

st.line_chart(df["Close"])

img_tensor = price_to_image(df)

with torch.no_grad():
    cnn_prob = model(img_tensor).item()

st.metric("CNN Volatility Spike Probability", round(cnn_prob, 3))

final_signal = 0.6 * sentiment_score + 0.4 * cnn_prob

st.metric("Composite Volatility Signal", round(final_signal, 3))

if final_signal > 0.3:
    st.error("High Volatility Risk")
elif final_signal < -0.3:
    st.success("Low Volatility Environment")
else:
    st.warning("Neutral Risk Regime")
