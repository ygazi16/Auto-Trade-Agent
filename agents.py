import asyncio
import alpaca_trade_api as tradeapi

# NewsAgent: gets sentiment score from news headlines using OpenAI or VADER
class NewsAgent:
    async def run(self, ticker):
        # fetch news via NewsAPI or scrape
        # analyze sentiment using VADER or OpenAI
        # For demo, return a static score
        return {"ticker": ticker, "sentiment_score": 0.7}

# DataAnalysisAgent: uses yfinance or Finnhub to get EPS, P/E, beta, volume and score them
class DataAnalysisAgent:
    async def run(self, ticker):
        # fetch metrics using yfinance
        # evaluate quality using basic scoring rules or ML
        # For demo, return a static score
        return {"ticker": ticker, "fundamental_score": 0.85}

# TrendAgent: computes MACD, RSI, Bollinger Bands to assess trend strength
class TrendAgent:
    async def run(self, ticker):
        # get historical prices
        # calculate technical indicators
        # output score between 0-1
        # For demo, return a static score
        return {"ticker": ticker, "trend_score": 0.8}

# AllocatorAgent: takes all 3 scores and decides whether to buy and how much
class AllocatorAgent:
    def run(self, scores):
        # weighted average of sentiment, fundamentals, trend
        confidence = (
            scores["sentiment_score"] +
            scores["fundamental_score"] +
            scores["trend_score"]
        ) / 3
        if confidence > 0.75:
            return {"action": "buy", "ticker": scores["ticker"], "amount": 500}
        return {"action": "hold"}

# ExecutionAgent using Alpaca paper trading API
class ExecutionAgent:
    def __init__(self, key, secret, paper=True):
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(key, secret, base_url, api_version='v2')

    def execute_trade(self, action, ticker, amount):
        if action == "buy":
            price = float(self.api.get_last_trade(ticker).price)
            qty = int(amount // price)
            self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )

# ControllerAgent to run NewsAgent, DataAgent, TrendAgent in parallel using asyncio
class ControllerAgent:
    def __init__(self, news, data, trend, allocator, executor):
        self.news = news
        self.data = data
        self.trend = trend
        self.allocator = allocator
        self.executor = executor

    async def run(self, ticker):
        n = asyncio.create_task(self.news.run(ticker))
        d = asyncio.create_task(self.data.run(ticker))
        t = asyncio.create_task(self.trend.run(ticker))

        news_result, data_result, trend_result = await asyncio.gather(n, d, t)
        combined = {
            "ticker": ticker,
            "sentiment_score": news_result["sentiment_score"],
            "fundamental_score": data_result["fundamental_score"],
            "trend_score": trend_result["trend_score"]
        }
        decision = self.allocator.run(combined)
        if decision["action"] == "buy":
            self.executor.execute_trade("buy", ticker, decision["amount"])
