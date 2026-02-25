import feedparser

def fetch_rss(url):
    feed = feedparser.parse(url)
    headlines = []
    for entry in feed.entries[:20]:
        headlines.append(entry.title)
    return headlines
