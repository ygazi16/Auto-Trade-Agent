import pickle
from agents import ControllerAgent, NewsAgent, DataAnalysisAgent, TrendAgent, AllocatorAgent, ExecutionAgent

# Load the trained MLAgent
with open('ml_agent_trained.pkl', 'rb') as f:
    ml_agent = pickle.load(f)

# Instantiate other agents (replace with your real API keys and executor setup)
news_agent = NewsAgent()
data_agent = DataAnalysisAgent()
trend_agent = TrendAgent()
allocator_agent = AllocatorAgent()
executor_agent = ExecutionAgent(key='YOUR_ALPACA_KEY', secret='bEAfWxDZF5xyyaCu8cnIPQdJhMMLDUKWtvUOISWz')

# Create the controller with MLAgent
controller = ControllerAgent(
    news=news_agent,
    data=data_agent,
    trend=trend_agent,
    allocator=allocator_agent,
    executor=executor_agent,
    ml_agent=ml_agent
)

ticker = 'AAPL'  # Example ticker

ticker = 'AAPL'  # Example ticker

import yfinance as yf
import asyncio
import pandas as pd
from datetime import datetime, timedelta

ticker = 'AAPL'  # Example ticker
initial_cash = 100000
cash = initial_cash
holding_qty = 0
shares_held = 0
trade_log = []

# Fetch historical prices for the simulation period
start_date = '2024-01-01'
end_date = '2025-01-01'
data = yf.Ticker(ticker).history(start=start_date, end=end_date)

async def main():
    global cash, holding_qty, shares_held
    for date, row in data.iterrows():
        price = float(row['Close'])
        extra_features = {'price': price}
        decision = await controller.run(ticker, holding_qty, extra_features)
        action = decision.get('action')
        amount = decision.get('amount', 0)
        # Simulate cash and holdings
        if action == 'buy' and cash >= amount:
            qty = int(amount // price)
            if qty > 0:
                cost = qty * price
                cash -= cost
                shares_held += qty
                holding_qty = shares_held
                trade_log.append((date.date(), 'buy', qty, price, cash, shares_held))
        elif action == 'sell' and shares_held > 0:
            qty = int(amount // price)
            qty = min(qty, shares_held)
            if qty > 0:
                proceeds = qty * price
                cash += proceeds
                shares_held -= qty
                holding_qty = shares_held
                trade_log.append((date.date(), 'sell', qty, price, cash, shares_held))
        # else hold
    # Final portfolio value
    final_price = float(data.iloc[-1]['Close'])
    portfolio_value = cash + shares_held * final_price
    print(f"Simulation complete for {ticker} from {start_date} to {end_date}")
    print(f"Final cash: ${cash:,.2f}")
    print(f"Final shares held: {shares_held}")
    print(f"Final price: ${final_price:,.2f}")
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    print(f"Return: {((portfolio_value/initial_cash)-1)*100:.2f}%")
    print("Trade log (date, action, qty, price, cash, shares held):")
    for log in trade_log:
        print(log)

if __name__ == '__main__':
    asyncio.run(main())
