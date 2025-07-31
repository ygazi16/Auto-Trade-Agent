
import asyncio
from agents import NewsAgent, DataAnalysisAgent, TrendAgent, AllocatorAgent, ControllerAgent
import yfinance as yf
import os
from dotenv import load_dotenv
import pandas as pd
import requests

# Load environment variables from .env file
load_dotenv()

# SimulatedExecutionAgent using yfinance historical data
class SimulatedExecutionAgent:
    def __init__(self, stop_loss_abs=None, take_profit_pct=0.15):
        self.trades = []
        self.holdings = 0
        self.cash = 0
        self.last_buy_price = None
        self.stop_loss_abs = stop_loss_abs  # absolute dollar stop-loss (e.g. 2*ATR)
        self.take_profit_pct = take_profit_pct  # e.g. 0.15 for 15% take-profit

    def execute_trade(self, action, ticker, amount, price, date):
        if action == "buy":
            qty = int(amount // price)
            if qty > 0:
                self.trades.append({
                    "date": date,
                    "ticker": ticker,
                    "qty": qty,
                    "price": price,
                    "total": qty * price,
                    "type": "buy"
                })
                self.holdings += qty
                self.cash -= qty * price
                self.last_buy_price = price  # Track last buy price for stop-loss/take-profit
                print(f"[SIMULATION] {date}: Buying {qty} shares of {ticker} at ${price:.2f} (Total: ${qty*price:.2f})")
        elif action == "sell" and self.holdings > 0:
            qty = min(amount, self.holdings)
            if qty > 0:
                self.trades.append({
                    "date": date,
                    "ticker": ticker,
                    "qty": qty,
                    "price": price,
                    "total": qty * price,
                    "type": "sell"
                })
                self.holdings -= qty
                self.cash += qty * price
                if self.holdings == 0:
                    self.last_buy_price = None
                print(f"[SIMULATION] {date}: Selling {qty} shares of {ticker} at ${price:.2f} (Total: ${qty*price:.2f})")
        else:
            print(f"[SIMULATION] {date}: No trade executed for {ticker}.")

    def check_stop_loss_take_profit(self, price):
        if self.last_buy_price is None or self.holdings == 0:
            return None
        # ATR-based stop-loss
        if self.stop_loss_abs is not None:
            stop_loss_price = self.last_buy_price - self.stop_loss_abs
        else:
            stop_loss_price = None
        take_profit_price = self.last_buy_price * (1 + self.take_profit_pct)
        if stop_loss_price is not None and price <= stop_loss_price:
            return "stop_loss"
        elif price >= take_profit_price:
            return "take_profit"
        return None


async def main():


    # Fetch S&P 500 tickers from Wikipedia
    print("Fetching S&P 500 tickers from Wikipedia...")
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp500_df = table[0]
    sp500_tickers = sp500_df['Symbol'].tolist()
    # Some tickers have "." instead of "-" for Yahoo Finance
    sp500_tickers = [t.replace('.', '-') for t in sp500_tickers]

    # Add top 50 most traded precious metals tickers (ETF/commodity proxies)
    metals_tickers = [
        'GC=F', 'SI=F', 'PL=F', 'PA=F', 'HG=F', 'ZNC=F', 'AL=F', 'CU=F', 'PALL', 'PPLT',
        'GLD', 'SLV', 'IAU', 'SGOL', 'SIVR', 'CPER', 'JJN', 'JJC', 'DBB', 'USLM',
        'NEM', 'GOLD', 'AEM', 'WPM', 'FNV', 'KGC', 'AU', 'SBSW', 'HL', 'CDE',
        'AG', 'EXK', 'FSM', 'MAG', 'MUX', 'PAAS', 'SILV', 'GATO', 'SAND', 'OR',
        'RGLD', 'WPM', 'SBSW', 'PLG', 'IVPAF', 'SBGL', 'SBSW', 'SBSW', 'SBSW', 'SBSW'
    ][:50]

    # Add 50 earth minerals market tickers (rare earth, lithium, uranium, etc. stocks/ETFs)
    earth_minerals_tickers = [
        'LIT', 'ALB', 'SQM', 'PLL', 'LAC', 'MP', 'PILBF', 'LTHM', 'SGML', 'CYDVF',
        'U=F', 'URA', 'CCJ', 'UEC', 'NXE', 'DNN', 'URG', 'UUUU', 'LEU', 'BOIL',
        'REMX', 'AVLNF', 'LYSCF', 'GDLNF', 'TAS', 'ILHMF', 'GDLNF', 'GDLNF', 'GDLNF', 'GDLNF',
        'TMRC', 'VTMXF', 'FMC', 'PWRMF', 'WLCDF', 'WMLLF', 'PEMIF', 'NLC', 'NMTLF', 'NMT.AX',
        'SYAAF', 'GXY.AX', 'OROCF', 'LKE.AX', 'LKE', 'LPI.AX', 'LPI', 'EMHLF', 'EMH', 'E25.AX'
    ][:50]


    # Use first 30 S&P 500 tickers for broader testing
    tickers = sp500_tickers[:30]

    # For asset class analysis
    asset_classes = {}
    for t in sp500_tickers:
        asset_classes[t] = 'S&P 500'
    for t in metals_tickers:
        asset_classes[t] = 'Precious Metals'
    for t in earth_minerals_tickers:
        asset_classes[t] = 'Earth Minerals'


    # === FIXED PARAMETER SIMULATION ===
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    rsi_buy = 40
    rsi_sell = 60
    macd_buy = -0.5  # Use the more inclusive value from previous grid
    macd_sell = 0    # Use the more inclusive value from previous grid
    tp = 0.15
    sl_mult = 1.5
    total_portfolio_profit = 0
    total_invested = 0
    print(f"\n=== Running simulation with fixed params: RSI<{rsi_buy}, RSI>{rsi_sell}, TP={tp}, SL={sl_mult}xATR ===")
    for ticker in tickers:
        print(f"\n=== Simulating {ticker} ===")
        news_agent = NewsAgent()
        data_agent = DataAnalysisAgent()
        trend_agent = TrendAgent()
        allocator_agent = AllocatorAgent()
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if data.empty:
            continue
        atr_period = 14
        if len(data) >= atr_period:
            high = data['High']
            low = data['Low']
            close = data['Close']
            tr = pd.concat([
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
            atr_value = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        else:
            atr_value = 0
        stop_loss_abs = sl_mult * atr_value if atr_value > 0 else None
        execution_agent = SimulatedExecutionAgent(stop_loss_abs=stop_loss_abs, take_profit_pct=tp)
        controller = ControllerAgent(
            news_agent,
            data_agent,
            trend_agent,
            allocator_agent,
            execution_agent
        )
        closes = data['Close']
        volumes = data['Volume']
        window_rsi = 14
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        bb_window = 20
        bb_ma = closes.rolling(window=bb_window).mean()
        bb_std = closes.rolling(window=bb_window).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        for idx, (date, row) in enumerate(data.iterrows()):
            price = float(row['Close'])
            if idx >= 5:
                momentum = (closes.iloc[idx] - closes.iloc[idx-5]) / closes.iloc[idx-5]
            else:
                momentum = 0
            if idx >= 5:
                volatility = closes.iloc[idx-5:idx].std() / closes.iloc[idx-5:idx].mean()
            else:
                volatility = 0
            if idx >= 5:
                volume = volumes.iloc[idx] / (volumes.iloc[idx-5:idx].mean() + 1e-9)
            else:
                volume = 1
            rsi_val = rsi.iloc[idx] if idx < len(rsi) else None
            macd_val = macd.iloc[idx] if idx < len(macd) else None
            macd_signal = signal.iloc[idx] if idx < len(signal) else None
            macd_hist_val = macd_hist.iloc[idx] if idx < len(macd_hist) else None
            bb_upper_val = bb_upper.iloc[idx] if idx < len(bb_upper) else None
            bb_lower_val = bb_lower.iloc[idx] if idx < len(bb_lower) else None
            bb_ma_val = bb_ma.iloc[idx] if idx < len(bb_ma) else None
            extra_features = {
                'momentum': momentum,
                'volatility': volatility,
                'volume': volume,
                'rsi': rsi_val,
                'macd': macd_val,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist_val,
                'bb_upper': bb_upper_val,
                'bb_lower': bb_lower_val,
                'bb_ma': bb_ma_val
            }
            stop_action = execution_agent.check_stop_loss_take_profit(price)
            if stop_action == "stop_loss":
                execution_agent.execute_trade("sell", ticker, execution_agent.holdings, price, date.date())
                continue
            elif stop_action == "take_profit":
                execution_agent.execute_trade("sell", ticker, execution_agent.holdings, price, date.date())
                continue
            # === LOOSER BUY/SELL FILTERS ===
            # Buy: RSI < rsi_buy, MACD >= macd_buy, price > BB middle, momentum > 0
            # Sell: RSI > rsi_sell, MACD <= macd_sell, price < BB middle, momentum < 0
            buy_signal = (
                rsi_val is not None and rsi_val < rsi_buy and
                macd_val is not None and macd_val >= macd_buy and
                price > (bb_ma_val if bb_ma_val is not None else price) and
                momentum > 0
            )
            sell_signal = (
                rsi_val is not None and rsi_val > rsi_sell and
                macd_val is not None and macd_val <= macd_sell and
                price < (bb_ma_val if bb_ma_val is not None else price) and
                momentum < 0
            )
            n = asyncio.create_task(news_agent.run(ticker))
            d = asyncio.create_task(data_agent.run(ticker))
            t_ = asyncio.create_task(trend_agent.run(ticker, momentum=momentum, volatility=volatility))
            news_result, data_result, trend_result = await asyncio.gather(n, d, t_)
            combined = {
                "ticker": ticker,
                "sentiment_score": news_result["sentiment_score"],
                "fundamental_score": data_result["fundamental_score"],
                "trend_score": trend_result["trend_score"]
            }
            base_amount = 10000
            min_amount = 1000
            scores = [combined['sentiment_score'], combined['fundamental_score'], combined['trend_score']]
            confidence = (sum(scores) / (3 * 100)) if all(isinstance(s, (int, float)) for s in scores) else 0.5
            vol_scale = 1.0 / (volatility + 1e-3)
            vol_scale = min(vol_scale, 3.0)
            amount = min(base_amount, max(min_amount, base_amount * confidence * vol_scale * 0.5))
            # Only execute buy/sell if strict filter passes
            if buy_signal:
                execution_agent.execute_trade("buy", ticker, amount, price, date.date())
            elif sell_signal:
                execution_agent.execute_trade("sell", ticker, execution_agent.holdings, price, date.date())
        invested = sum(trade['qty'] * trade['price'] for trade in execution_agent.trades if trade['type'] == 'buy')
        if not data.empty:
            last_close = float(data['Close'].iloc[-1])
            unrealized = execution_agent.holdings * last_close
            total_profit = execution_agent.cash + unrealized
            total_portfolio_profit += total_profit
            total_invested += invested
    print(f"\nTotal portfolio profit: ${total_portfolio_profit:.2f} (Invested: ${total_invested:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
