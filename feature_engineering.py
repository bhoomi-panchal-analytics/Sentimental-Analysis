# feature_engineering.py

import numpy as np
import pandas as pd

def create_features(df):

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["realized_vol"] = df["log_return"].rolling(5).std() * np.sqrt(252)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["momentum"] = df["Close"].pct_change(5)

    df.dropna(inplace=True)

    return df


def make_sequences(data, seq_len, feature_cols):
    X = []

    for i in range(seq_len, len(data)):
        X.append(data[feature_cols].iloc[i-seq_len:i].values)

    return np.array(X)
