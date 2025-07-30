import asyncio
from agents import NewsAgent, DataAnalysisAgent, TrendAgent, AllocatorAgent, ExecutionAgent, ControllerAgent
import os

# Set your Alpaca API credentials here or use environment variables
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_ALPACA_SECRET_KEY')

async def main():
    ticker = "AAPL"  # Example ticker

    # Instantiate agents
    news_agent = NewsAgent()
    data_agent = DataAnalysisAgent()
    trend_agent = TrendAgent()
    allocator_agent = AllocatorAgent()
    execution_agent = ExecutionAgent(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    # Create controller
    controller = ControllerAgent(
        news_agent,
        data_agent,
        trend_agent,
        allocator_agent,
        execution_agent
    )

    # Run the controller for the ticker
    await controller.run(ticker)
    print(f"Completed trading decision for {ticker}.")

if __name__ == "__main__":
    asyncio.run(main())
