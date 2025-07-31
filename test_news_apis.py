import os
from datetime import datetime, timedelta
from agents import NewsAgent

if __name__ == "__main__":
    agent = NewsAgent()
    ticker = "AAPL"
    until_date = datetime.utcnow().date()
    from_param = (until_date - timedelta(days=7)).isoformat()
    to_param = until_date.isoformat()
    results = {}
    results['newsapi'] = agent._fetch_newsapi(ticker, from_param, to_param)
    results['gnews'] = agent._fetch_gnews(ticker, from_param, to_param)
    results['contextualweb'] = agent._fetch_contextualweb(ticker, from_param, to_param)
    results['bing'] = agent._fetch_bing(ticker, from_param, to_param)
    results['fmp'] = agent._fetch_fmp(ticker, from_param, to_param)
    results['alphavantage'] = agent._fetch_alphavantage(ticker, from_param, to_param)
    results['nyt'] = agent._fetch_nyt(ticker, from_param, to_param)
    results['guardian'] = agent._fetch_guardian(ticker, from_param, to_param)
    results['eventregistry'] = agent._fetch_eventregistry(ticker, from_param, to_param)
    for k, v in results.items():
        print(f"{k}: {len(v)} headlines fetched.")
        if v:
            print(f"  Example: {v[0]}")
        else:
            print("  No headlines fetched or API key missing.")
