import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_crypto_news(currencies=None, filter=None, limit=50):
    """
    Fetch news from CryptoPanic API
    
    Parameters:
    - currencies: list of currency symbols (e.g., ['BTC', 'ETH'])
    - filter: str ('rising'|'hot'|'bullish'|'bearish'|'important'|'saved')
    - limit: int, number of results (default 50)
    """
    api_key = os.getenv('CRYPTOPANIC_API_KEY')
    if not api_key:
        raise ValueError("Missing CRYPTOPANIC_API_KEY in .env file")

    base_url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        'auth_token': api_key,
        'limit': limit,
        'public': 'true'
    }
    
    if currencies:
        params['currencies'] = ','.join(currencies)
    if filter:
        params['filter'] = filter

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def analyze_news(data):
    """Convert API response to DataFrame and analyze sentiment distribution"""
    if not data or 'results' not in data:
        return None
    
    # Extract relevant information
    news_items = []
    for item in data['results']:
        news_item = {
            'title': item['title'],
            'published_at': item['published_at'],
            'sentiment': item.get('votes', {}).get('sentiment', 'neutral'),
            'url': item['url'],
            'currencies': [c['code'] for c in item.get('currencies', [])],
            'source': item['source']['title']
        }
        news_items.append(news_item)
    
    # Convert to DataFrame
    df = pd.DataFrame(news_items)
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    return df

if __name__ == "__main__":
    # Example usage
    currencies = ['BTC', 'ETH']  # Filter for Bitcoin and Ethereum news
    
    # Fetch news
    print(f"Fetching latest crypto news for {', '.join(currencies)}...")
    news_data = fetch_crypto_news(currencies=currencies, filter='hot', limit=50)
    
    if news_data:
        # Analyze and display results
        df = analyze_news(news_data)
        if df is not None:
            print("\nLatest news summary:")
            print(f"Total articles: {len(df)}")
            
            # Display sentiment distribution
            print("\nSentiment distribution:")
            print(df['sentiment'].value_counts())
            
            # Display latest news
            print("\nLatest 5 articles:")
            latest = df.sort_values('published_at', ascending=False).head()
            for _, row in latest.iterrows():
                print(f"\nTitle: {row['title']}")
                print(f"Published: {row['published_at']}")
                print(f"Sentiment: {row['sentiment']}")
                print(f"Source: {row['source']}")
                print(f"URL: {row['url']}")
                print("-" * 80)
            
            # Save to CSV
            filename = f"crypto_news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False)
            print(f"\nData saved to {filename}")
