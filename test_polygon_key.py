import os
import requests
from dotenv import load_dotenv

load_dotenv()
polygon_key = os.getenv('POLYGON_KEY')

ticker = 'AAPL'
url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={polygon_key}"
resp = requests.get(url)
print('Status code:', resp.status_code)
print('Response:', resp.json())
