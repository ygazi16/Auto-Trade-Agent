from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
# MLAgent: trains and predicts buy/sell/hold using scikit-learn
class MLAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, features):
        # Extracts a fixed feature vector from the input dict
        # Use the same features as MultiFactorScorer plus technicals
        keys = [
            'pe', 'pb', 'dividend_yield',
            'revenue_growth', 'eps_growth', 'analyst_target_price', 'price',
            'momentum', 'rsi', 'macd',
            'roe', 'roa', 'profit_margin', 'debt_equity', 'free_cash_flow',
            'volatility', 'volume', 'beta', 'esg_score',
        ]
        vec = []
        for k in keys:
            v = features.get(k, 0)
            try:
                v = float(v)
            except Exception:
                v = 0
            vec.append(v)
        return np.array(vec)

    def fit(self, X_dicts, y):
        # X_dicts: list of feature dicts, y: list of labels (0=sell, 1=hold, 2=buy)
        X = np.array([self.extract_features(f) for f in X_dicts])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, features):
        # features: dict
        X = self.extract_features(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        # Return action and confidence
        action = {0: 'sell', 1: 'hold', 2: 'buy'}.get(pred, 'hold')
        return {'action': action, 'confidence': float(np.max(proba))}
# MultiFactorScorer: combines value, growth, momentum, quality into a composite score
class MultiFactorScorer:
    def score(self, features):
        # Value: low P/E, high dividend yield, low P/B
        pe = features.get('pe', None)
        pb = features.get('pb', None)
        dividend_yield = features.get('dividend_yield', None)
        value_score = 0.5
        if pe is not None:
            value_score += 0.2 * (1 - min(max(float(pe)/30, 0), 1))
        if pb is not None:
            value_score += 0.1 * (1 - min(max(float(pb)/5, 0), 1))
        if dividend_yield is not None:
            value_score += 0.1 * min(max(float(dividend_yield)/0.05, 0), 1)

        # Growth: revenue/EPS growth, analyst target price
        revenue_growth = features.get('revenue_growth', None)
        eps_growth = features.get('eps_growth', None)
        analyst_target_price = features.get('analyst_target_price', None)
        price = features.get('price', None)
        growth_score = 0.5
        if revenue_growth is not None:
            growth_score += 0.2 * min(max(float(revenue_growth)/0.2, 0), 1)
        if eps_growth is not None:
            growth_score += 0.2 * min(max(float(eps_growth)/0.2, 0), 1)
        if analyst_target_price is not None and price is not None:
            try:
                upside = (float(analyst_target_price) - float(price)) / float(price)
                growth_score += 0.1 * min(max(upside/0.2, 0), 1)
            except Exception:
                pass

        # Momentum: price momentum, RSI, MACD
        momentum = features.get('momentum', 0)
        rsi = features.get('rsi', 50)
        macd = features.get('macd', 0)
        momentum_score = 0.5 + 0.2 * momentum
        if rsi < 30:
            momentum_score -= 0.1
        elif rsi > 70:
            momentum_score += 0.1
        momentum_score += 0.1 * (1 if macd > 0 else -1)

        # Quality: ROE, ROA, profit margin, debt/equity, cash flow
        roe = features.get('roe', None)
        roa = features.get('roa', None)
        profit_margin = features.get('profit_margin', None)
        debt_equity = features.get('debt_equity', None)
        free_cf = features.get('free_cash_flow', None)
        quality_score = 0.5
        if roe is not None:
            quality_score += 0.1 * min(max(float(roe)/0.3, 0), 1)
        if roa is not None:
            quality_score += 0.1 * min(max(float(roa)/0.15, 0), 1)
        if profit_margin is not None:
            quality_score += 0.1 * min(max(float(profit_margin)/0.3, 0), 1)
        if debt_equity is not None:
            quality_score += 0.1 * (1 - min(max(float(debt_equity)/2, 0), 1))
        if free_cf is not None:
            quality_score += 0.1 * min(max(float(free_cf)/1e10, 0), 1)

        # Composite
        composite = (value_score + growth_score + momentum_score + quality_score) / 4
        return {
            'value': value_score,
            'growth': growth_score,
            'momentum': momentum_score,
            'quality': quality_score,
            'composite': composite
        }

# EventAgent: fetches and reacts to earnings, splits, insider trades, macro events
class EventAgent:
    def __init__(self, fmp_key):
        self.fmp_key = fmp_key

    def get_earnings(self, ticker):
        url = f"https://financialmodelingprep.com/api/v3/earning_calendar/{ticker}?limit=1&apikey={self.fmp_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
        except Exception:
            pass
        return None

    def get_splits(self, ticker):
        url = f"https://financialmodelingprep.com/api/v3/stock_split_calendar?symbol={ticker}&apikey={self.fmp_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
        except Exception:
            pass
        return None

    def get_insider_trades(self, ticker):
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&apikey={self.fmp_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
        except Exception:
            pass
        return None

import asyncio
import alpaca_trade_api as tradeapi
import random
import requests
import os
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# NewsAgent: gets sentiment score from newsapi.org headlines using VADER
class NewsAgent:
    def __init__(self):
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.gnews_key = os.getenv('GNEWS_KEY')
        self.contextualweb_key = os.getenv('CONTEXTUALWEB_KEY')
        self.bing_key = os.getenv('BING_KEY')
        self.fmp_key = os.getenv('FMP_KEY')
        self.alphavantage_key = os.getenv('ALPHAVANTAGE_KEY')
        self.nyt_key = os.getenv('NYT_KEY')
        self.guardian_key = os.getenv('GUARDIAN_KEY')
        self.eventregistry_key = os.getenv('EVENTREGISTRY_KEY')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def _fetch_newsapi(self, ticker, from_param, to_param):
        if not self.newsapi_key:
            return []
        url = (
            f"https://newsapi.org/v2/everything?q={ticker}&from={from_param}&to={to_param}"
            f"&sortBy=publishedAt&language=en&apiKey={self.newsapi_key}"
        )
        try:
            resp = requests.get(url)
            data = resp.json()
            return [a['title'] for a in data.get('articles', []) if 'title' in a]
        except Exception:
            return []

    def _fetch_gnews(self, ticker, from_param, to_param):
        if not self.gnews_key:
            return []
        url = f"https://gnews.io/api/v4/search?q={ticker}&from={from_param}&to={to_param}&lang=en&token={self.gnews_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            return [a['title'] for a in data.get('articles', []) if 'title' in a]
        except Exception:
            return []

    def _fetch_contextualweb(self, ticker, from_param, to_param):
        if not self.contextualweb_key:
            return []
        url = f"https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI?q={ticker}&fromPublishedDate={from_param}&toPublishedDate={to_param}&pageNumber=1&pageSize=25&autoCorrect=true"
        headers = {"X-RapidAPI-Key": self.contextualweb_key}
        try:
            resp = requests.get(url, headers=headers)
            data = resp.json()
            return [a['title'] for a in data.get('value', []) if 'title' in a]
        except Exception:
            return []

    def _fetch_bing(self, ticker, from_param, to_param):
        if not self.bing_key:
            return []
        url = f"https://api.bing.microsoft.com/v7.0/news/search?q={ticker}&freshness=Week&count=25"
        headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
        try:
            resp = requests.get(url, headers=headers)
            data = resp.json()
            return [a['name'] for a in data.get('value', []) if 'name' in a]
        except Exception:
            return []

    def _fetch_fmp(self, ticker, from_param, to_param):
        if not self.fmp_key:
            return []
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=25&apikey={self.fmp_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            return [a['title'] for a in data if 'title' in a]
        except Exception:
            return []

    def _fetch_alphavantage(self, ticker, from_param, to_param):
        if not self.alphavantage_key:
            return []
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.alphavantage_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            return [a['title'] for a in data.get('feed', []) if 'title' in a]
        except Exception:
            return []

    def _fetch_nyt(self, ticker, from_param, to_param):
        if not self.nyt_key:
            return []
        url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={ticker}&begin_date={from_param.replace('-','')}&end_date={to_param.replace('-','')}&api-key={self.nyt_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            return [a['headline']['main'] for a in data.get('response', {}).get('docs', []) if 'headline' in a and 'main' in a['headline']]
        except Exception:
            return []

    def _fetch_guardian(self, ticker, from_param, to_param):
        if not self.guardian_key:
            return []
        url = f"https://content.guardianapis.com/search?q={ticker}&from-date={from_param}&to-date={to_param}&api-key={self.guardian_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            return [a['webTitle'] for a in data.get('response', {}).get('results', []) if 'webTitle' in a]
        except Exception:
            return []

    def _fetch_eventregistry(self, ticker, from_param, to_param):
        if not self.eventregistry_key:
            return []
        url = f"https://eventregistry.org/api/v1/article/getArticles?keyword={ticker}&dateStart={from_param}&dateEnd={to_param}&apiKey={self.eventregistry_key}"
        try:
            resp = requests.get(url)
            data = resp.json()
            return [a['title'] for a in data.get('articles', {}).get('results', []) if 'title' in a]
        except Exception:
            return []

    async def run(self, ticker, until_date=None):
        # Aggregate news from all APIs for the ticker up to until_date
        if until_date is None:
            until_date = datetime.utcnow().date()
        if isinstance(until_date, datetime):
            until_date = until_date.date()
        from_param = (until_date - timedelta(days=7)).isoformat()
        to_param = until_date.isoformat()

        headlines = set()
        # Fetch from all APIs
        headlines.update(self._fetch_newsapi(ticker, from_param, to_param))
        headlines.update(self._fetch_gnews(ticker, from_param, to_param))
        headlines.update(self._fetch_contextualweb(ticker, from_param, to_param))
        headlines.update(self._fetch_bing(ticker, from_param, to_param))
        headlines.update(self._fetch_fmp(ticker, from_param, to_param))
        headlines.update(self._fetch_alphavantage(ticker, from_param, to_param))
        headlines.update(self._fetch_nyt(ticker, from_param, to_param))
        headlines.update(self._fetch_guardian(ticker, from_param, to_param))
        headlines.update(self._fetch_eventregistry(ticker, from_param, to_param))

        # Filter out None or empty headlines
        filtered_headlines = [h for h in headlines if h and isinstance(h, str) and h.strip()]
        if not filtered_headlines:
            sentiment_score = 0.5  # Neutral if no news
        else:
            scores = [self.sentiment_analyzer.polarity_scores(h)['compound'] for h in filtered_headlines]
            sentiment_score = (sum(scores) / len(scores) + 1) / 2
        return {"ticker": ticker, "sentiment_score": sentiment_score}

# DataAnalysisAgent: uses yfinance or Finnhub to get EPS, P/E, beta, volume and score them
class DataAnalysisAgent:
    def __init__(self):
        self.polygon_key = os.getenv('POLYGON_KEY')

    async def run(self, ticker):
        # Use Polygon.io to get profile, financials, and analyst data
        if not self.polygon_key:
            fundamental_score = random.uniform(0.5, 0.95)
            return {"ticker": ticker, "fundamental_score": fundamental_score}
        try:
            # Company profile
            url_profile = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={self.polygon_key}"
            resp_profile = requests.get(url_profile)
            data_profile = resp_profile.json().get('results', {})
            sector = data_profile.get('sic_description', None)
            industry = data_profile.get('market', None)
            # Financials (latest annual)
            url_financials = f"https://api.polygon.io/vX/reference/financials?ticker={ticker}&limit=1&type=annual&apiKey={self.polygon_key}"
            resp_fin = requests.get(url_financials)
            data_fin = resp_fin.json().get('results', [{}])[0]
            revenue = data_fin.get('revenue', None)
            eps = data_fin.get('earningsPerBasicShare', None)
            total_assets = data_fin.get('assets', None)
            total_liabilities = data_fin.get('liabilities', None)
            cash = data_fin.get('cashAndCashEquivalents', None)
            operating_cf = data_fin.get('operatingCashFlow', None)
            free_cf = data_fin.get('freeCashFlow', None)
            roe = data_fin.get('roe', None)
            roa = data_fin.get('roa', None)
            debt_equity = data_fin.get('debtEquityRatio', None)
            # Analyst ratings (not available in Polygon, set to None)
            analyst_rating = None
            analyst_target_price = None
            # Score: combine all available metrics
            score = 0.5
            try:
                eps_score = min(max(float(eps)/10, 0), 1) if eps is not None else 0.5
                revenue_score = min(max(float(revenue)/1e11, 0), 1) if revenue is not None else 0.5
                roe_score = min(max(float(roe)/0.5, 0), 1) if roe is not None else 0.5
                analyst_score = 0.5  # No analyst rating
                score = (eps_score + revenue_score + roe_score + analyst_score) / 4
            except Exception:
                pass
            return {
                "ticker": ticker,
                "fundamental_score": score,
                "eps": eps,
                "revenue": revenue,
                "sector": sector,
                "industry": industry,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "cash": cash,
                "operating_cash_flow": operating_cf,
                "free_cash_flow": free_cf,
                "roe": roe,
                "roa": roa,
                "debt_equity": debt_equity,
                "analyst_rating": analyst_rating,
                "analyst_target_price": analyst_target_price
            }
        except Exception:
            return {"ticker": ticker, "fundamental_score": 0.5}

# TrendAgent: computes MACD, RSI, Bollinger Bands to assess trend strength
class TrendAgent:
    async def run(self, ticker, momentum=0, volatility=0):
        # Make trend score dynamic based on momentum and volatility
        base = 0.5 + 0.4 * momentum - 0.2 * volatility
        trend_score = min(max(base + random.uniform(-0.1, 0.1), 0), 1)
        return {"ticker": ticker, "trend_score": trend_score}

# AllocatorAgent: takes all 3 scores and decides whether to buy and how much
class AllocatorAgent:
    def __init__(self):
        self.holdings = 0

    def run(self, scores, holding_qty, extra_features=None):
        # weighted average of sentiment, fundamentals, trend
        confidence = (
            scores["sentiment_score"] +
            scores["fundamental_score"] +
            scores["trend_score"]
        ) / 3

        # Use price momentum, volatility, volume, and FMP features in decision
        momentum = extra_features.get('momentum', 0) if extra_features else 0
        volatility = extra_features.get('volatility', 0) if extra_features else 0
        volume = extra_features.get('volume', 0) if extra_features else 0
        sector = extra_features.get('sector', None) if extra_features else None
        industry = extra_features.get('industry', None) if extra_features else None
        analyst_rating = extra_features.get('analyst_rating', None) if extra_features else None
        analyst_target_price = extra_features.get('analyst_target_price', None) if extra_features else None
        revenue = extra_features.get('revenue', None) if extra_features else None

        # Adjust confidence based on features
        confidence += 0.1 * momentum
        confidence -= 0.05 * volatility
        confidence += 0.05 * volume

        # Analyst rating boost
        if analyst_rating and isinstance(analyst_rating, str):
            if analyst_rating.lower() in ['buy', 'strong buy']:
                confidence += 0.05
            elif analyst_rating.lower() in ['sell', 'strong sell']:
                confidence -= 0.05

        # Revenue boost for large companies
        if revenue is not None:
            try:
                revenue_val = float(revenue)
                if revenue_val > 1e10:
                    confidence += 0.05
            except Exception:
                pass

        # Example: sector/industry-based adjustment (customize as needed)
        if sector and sector.lower() == 'technology':
            confidence += 0.02
        if industry and 'semiconductor' in industry.lower():
            confidence += 0.02

        # Lower buy threshold to 0.65, raise sell threshold to 0.6
        if confidence > 0.65:
            return {"action": "buy", "ticker": scores["ticker"], "amount": 500}
        elif confidence < 0.6 and holding_qty > 0:
            return {"action": "sell", "ticker": scores["ticker"], "amount": holding_qty}
        return {"action": "hold"}

# ExecutionAgent using Alpaca paper trading API
class ExecutionAgent:
    def __init__(self, key, secret, paper=True):
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(key, secret, base_url, api_version='v2')

    def execute_trade(self, action, ticker, amount, price=None):
        # Use the provided price for order sizing; do not fetch from Alpaca data API
        if price is None:
            raise ValueError("Price must be provided for trade execution.")
        qty = int(amount // price)
        if qty < 1:
            print(f"Not enough funds to execute {action} for {ticker} at price {price}")
            return
        if action == "buy":
            self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif action == "sell":
            self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )

# ControllerAgent to run NewsAgent, DataAgent, TrendAgent in parallel using asyncio
class ControllerAgent:
    def __init__(self, news, data, trend, allocator, executor, ml_agent=None):
        self.news = news
        self.data = data
        self.trend = trend
        self.allocator = allocator
        self.executor = executor
        self.ml_agent = ml_agent  # Optional MLAgent

    async def run(self, ticker, holding_qty=0, extra_features=None):
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
        # Merge all features for MLAgent
        features = dict(combined)
        if extra_features:
            features.update(extra_features)
        if isinstance(data_result, dict):
            features.update(data_result)
        # Use MLAgent if available and trained
        if self.ml_agent is not None and getattr(self.ml_agent, 'is_trained', False):
            ml_decision = self.ml_agent.predict(features)
            action = ml_decision['action']
            confidence = ml_decision['confidence']
            amount = 500 if action == 'buy' else holding_qty if action == 'sell' else 0
            decision = {"action": action, "ticker": ticker, "amount": amount, "ml_confidence": confidence}
        else:
            # Fallback to AllocatorAgent
            decision = self.allocator.run(combined, holding_qty, extra_features=features)
        # Use price from features or data_result
        price = features.get('price') or features.get('close')
        if decision["action"] == "buy":
            self.executor.execute_trade("buy", ticker, decision["amount"], price=price)
        elif decision["action"] == "sell":
            self.executor.execute_trade("sell", ticker, decision["amount"], price=price)
        return decision
