import pickle
from agents import MLAgent

# Example: Load your historical feature dicts and labels here
# X_dicts = [dict_of_features1, dict_of_features2, ...]
# y = [0, 2, 1, ...]  # 0=sell, 1=hold, 2=buy

# For demonstration, we'll use random data. Replace with your real data!
import random
X_dicts = []
y = []
for _ in range(200):
    features = {k: random.uniform(-1, 1) for k in [
        'pe', 'pb', 'dividend_yield', 'revenue_growth', 'eps_growth', 'analyst_target_price', 'price',
        'momentum', 'rsi', 'macd', 'roe', 'roa', 'profit_margin', 'debt_equity', 'free_cash_flow',
        'volatility', 'volume', 'beta', 'esg_score']}
    X_dicts.append(features)
    y.append(random.choice([0, 1, 2]))

ml_agent = MLAgent()
ml_agent.fit(X_dicts, y)

# Save the trained model for later use
with open('ml_agent_trained.pkl', 'wb') as f:
    pickle.dump(ml_agent, f)

print('MLAgent trained and saved as ml_agent_trained.pkl')
