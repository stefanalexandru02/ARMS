import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from datetime import timezone

load_dotenv()

def fetch_historical_news(currencies=None, start_date=None, end_date=None, filter=None, max_pages=10):
    """
    Fetch historical news data from CryptoPanic API with pagination
    """
    api_key = os.getenv('CRYPTOPANIC_API_KEY')
    if not api_key:
        raise ValueError("Missing CRYPTOPANIC_API_KEY in .env file")

    # Convert dates to UTC if they aren't timezone aware
    if start_date and start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    base_url = "https://cryptopanic.com/api/v1/posts/"
    all_results = []
    next_page = None
    page_count = 1

    while True:
        params = {
            'auth_token': api_key,
            'public': 'true',
            'limit': 100  # Maximum allowed per request
        }

        if currencies:
            params['currencies'] = ','.join(currencies)
        if filter:
            params['filter'] = filter
        if next_page:
            params['page'] = next_page

        try:
            print(f"Fetching page {page_count}...")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Filter results by date
            filtered_results = []
            for item in data['results']:
                # Parse ISO format date and ensure UTC timezone
                item_date = datetime.fromisoformat(item['published_at'].replace('Z', '+00:00'))
                if start_date and item_date < start_date:
                    return all_results  # We've gone past our start date
                if end_date and item_date > end_date:
                    continue
                filtered_results.append(item)

            all_results.extend(filtered_results)

            if not filtered_results or len(filtered_results) == 0:
                print("No more results to fetch.")
                break

            # Check if we should continue to next page
            next_page = data.get('next')
            if not next_page or page_count >= max_pages:
                break

            page_count += 1
            # Rate limiting - 1 request per second            

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
        time.sleep(3)

    return all_results

def process_historical_data(data):
    """Process and analyze historical news data"""
    news_items = []
    for item in data:
        news_item = {
            'title': item['title'],
            'published_at': datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')),
            'sentiment': item.get('votes', {}).get('sentiment', 'neutral'),
            'url': item['url'],
            'currencies': [c['code'] for c in item.get('currencies', [])],
            'source': item['source']['title']
        }
        news_items.append(news_item)
    
    df = pd.DataFrame(news_items)
    return df

def run(start_date, end_date, max_pages=10):
    currencies = ['BTC', 'ETH']

    print(f"Fetching historical crypto news from {start_date.date()} to {end_date.date()}")
    print(f"Currencies: {', '.join(currencies)}")
    
    # Fetch historical data
    historical_data = fetch_historical_news(
        currencies=currencies,
        start_date=start_date,
        end_date=end_date,
        filter='hot',
        max_pages=max_pages
    )
    
    if historical_data:
        # Process and analyze data
        df = process_historical_data(historical_data)
        
        # Basic analysis
        print(f"\nTotal articles collected: {len(df)}")
        
        # Daily article counts
        daily_counts = df.set_index('published_at').resample('D').size()
        print("\nDaily article counts:")
        print(daily_counts)
        
        # Sentiment distribution
        print("\nSentiment distribution:")
        print(df['sentiment'].value_counts(normalize=True) * 100)
        
        # Save to CSV with timestamp
        filename = f"historical_crypto_news_{start_date}.csv"
        print(f"\nSaving data to {filename}...")
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")

        # Sleep for 5 seconds before finishing
        time.sleep(5)
        
        # Optional: Save daily counts
        # daily_counts.to_csv(f"daily_counts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

if __name__ == "__main__":        
    # Set date range for last 30 days with timezone awareness
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    run(start_date, end_date)
