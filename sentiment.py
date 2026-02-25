from transformers import pipeline
import numpy as np

finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    return_all_scores=False
)

def aggregate_sentiment(headlines):
    scores = []
    for h in headlines:
        result = finbert(h[:512])[0]
        label = result['label']
        score = result['score']

        if label == "negative":
            scores.append(-score)
        elif label == "positive":
            scores.append(score)
        else:
            scores.append(0)

    return np.mean(scores)
