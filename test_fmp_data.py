
import os
from dotenv import load_dotenv
load_dotenv()
import requests

def test_fmp_data():
    fmp_key = os.getenv('FMP_KEY')
    ticker = "MSFT"
    print(f"Testing FMP endpoints for {ticker}\n")
    endpoints = {
        'profile': f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_key}",
        'income-statement': f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=1&apikey={fmp_key}",
        'rating': f"https://financialmodelingprep.com/api/v3/rating/{ticker}?apikey={fmp_key}"
    }
    for name, url in endpoints.items():
        print(f"Endpoint: {name}")
        resp = requests.get(url)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}\n")

if __name__ == "__main__":
    test_fmp_data()

if __name__ == "__main__":
    test_fmp_data()
