import yfinance as yf

def get_market_data(ticker="^VIX", period="5d", interval="5m"):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df
