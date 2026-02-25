# inference_engine.py

import torch
from feature_engineering import create_features, make_sequences
import numpy as np

def run_lstm_prediction(df, model, seq_len=20):

    df = create_features(df)

    feature_cols = ["log_return", "realized_vol", "ma_5", "ma_20", "momentum"]

    sequences = make_sequences(df, seq_len, feature_cols)

    if len(sequences) == 0:
        return None

    last_seq = torch.FloatTensor(sequences[-1]).unsqueeze(0)

    with torch.no_grad():
        prediction = model(last_seq).item()

    return prediction
