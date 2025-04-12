import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import json
from datetime import datetime, timedelta
import ast
from dotenv import load_dotenv
import matplotlib.dates as mdates

# Helper function to find project root
def PROJECT_ROOT():
    """Return the project root folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(symbol, interval):
    # Load environment variables from .env file
    load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT(), ".config/.env"))

    """
    Load all CSV files for a given symbol and interval, and concatenate them into a single DataFrame.
    
    :param symbol: The trading pair, e.g., BTCUSDT
    :param interval: The timeframe, e.g., 1m, 5m, 1h
    :return: A concatenated DataFrame containing all data.
    """
    data_path = os.path.join(PROJECT_ROOT(), '../data')
    if not data_path.startswith('/'):
        data_path = os.path.join('..', data_path)
    data_path = os.path.abspath(data_path)  # Ensure it's an absolute path
    print(f"Data path: {data_path}")
    file_pattern = os.path.join(data_path, f"{symbol}_{interval}", f"{symbol}_{interval}_*.csv")
    files = glob(file_pattern)

    # Load and concatenate all CSV files
    df = pd.concat([pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp') for file in files])
    df.sort_index(inplace=True)
    print(f"Loaded {len(df)} rows of data for {symbol} with interval {interval}")
    return df

def load_news_data():
    """
    Load the crypto news data
    
    :return: DataFrame containing news articles
    """
    data_path = os.path.join(PROJECT_ROOT(), '../data')
    if not data_path.startswith('/'):
        data_path = os.path.join('..', data_path)
    data_path = os.path.abspath(data_path)  # Ensure it's an absolute path
    
    news_file = os.path.join(data_path, "cryptonews.csv")
    news_df = pd.read_csv(news_file, parse_dates=['date'])
    print(f"Loaded {len(news_df)} news articles")
    
    # Parse sentiment dictionary from string to actual dictionary
    news_df['sentiment_dict'] = news_df['sentiment'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Extract sentiment components
    news_df['sentiment_class'] = news_df['sentiment_dict'].apply(lambda x: x.get('class', None) if isinstance(x, dict) else None)
    news_df['polarity'] = news_df['sentiment_dict'].apply(lambda x: x.get('polarity', None) if isinstance(x, dict) else None)
    news_df['subjectivity'] = news_df['sentiment_dict'].apply(lambda x: x.get('subjectivity', None) if isinstance(x, dict) else None)
    
    return news_df

def analyze_price_sentiment_relation(btc_df, news_df):
    """
    Analyze the relationship between price changes and news sentiment
    
    :param btc_df: DataFrame with Bitcoin price data
    :param news_df: DataFrame with news articles
    :return: None (plots and prints results)
    """
    # Add price change column to Bitcoin data
    btc_df['price_change'] = btc_df['close'].pct_change() * 100
    btc_df['price_change_direction'] = btc_df['price_change'].apply(lambda x: 'increase' if x > 0 else 'decrease')
    
    # Merge news and price data by date
    # Convert the date index of btc_df to a DatetimeIndex with just the date portion
    btc_df['date'] = btc_df.index.date
    
    # Convert news_df['date'] to just the date portion
    news_df['date'] = news_df['date'].dt.date
    
    # Group news articles by date and calculate average sentiment metrics
    daily_news = news_df.groupby('date').agg({
        'polarity': 'mean',
        'subjectivity': 'mean',
        'sentiment_class': lambda x: x.mode()[0] if not x.mode().empty else None
    }).reset_index()
    
    # Count articles by sentiment class per day
    sentiment_counts = pd.crosstab(news_df['date'], news_df['sentiment_class'])
    daily_news = daily_news.merge(sentiment_counts, left_on='date', right_index=True, how='left')
    
    # Fill NaN with 0 for counts
    for col in ['positive', 'negative', 'neutral']:
        if col in daily_news.columns:
            daily_news[col] = daily_news[col].fillna(0)
    
    # Now merge with btc_df
    merged_data = pd.merge(btc_df, daily_news, on='date', how='left')
    
    # Calculate correlations
    correlation = merged_data[['price_change', 'polarity', 'subjectivity']].corr()
    print("\nCorrelation between price change and sentiment metrics:")
    print(correlation)
    
    # Check if price increases/decreases are associated with different sentiment distributions
    price_up_sentiment = merged_data[merged_data['price_change'] > 0]['polarity'].mean()
    price_down_sentiment = merged_data[merged_data['price_change'] < 0]['polarity'].mean()
    
    print(f"\nAverage sentiment polarity on days with price increase: {price_up_sentiment:.4f}")
    print(f"Average sentiment polarity on days with price decrease: {price_down_sentiment:.4f}")
    
    # Plot price change vs. sentiment polarity
    plt.figure(figsize=(12, 6))
    plt.scatter(merged_data['polarity'], merged_data['price_change'], alpha=0.5)
    plt.title('Bitcoin Price Change vs. News Sentiment Polarity')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Price Change (%)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.savefig('price_vs_sentiment_polarity.png')
    
    # Filter data to only include days where we have both sentiment and price data
    valid_data = merged_data.dropna(subset=['polarity', 'close'])
    
    # Check if we have valid data
    if len(valid_data) == 0:
        print("Warning: No days found with both sentiment and price data")
        return
    
    print(f"\nAnalyzing {len(valid_data)} days that have both sentiment and price data")
    
    # Sort by date for proper time series visualization
    valid_data = valid_data.sort_values('date')
    
    # Plot price and sentiment over time - IMPROVED VERSION
    plt.figure(figsize=(14, 10))
    
    # Get date range - Fix: use the date column instead of index
    min_date = valid_data['date'].min()
    max_date = valid_data['date'].max()
    date_range = f"{min_date} to {max_date}"
    
    # Plot distribution of data availability
    ax0 = plt.subplot(3, 1, 1)
    ax0.plot(merged_data['date'], merged_data['close'].notnull().astype(int), 'bo', alpha=0.3, label='Price Data Available')
    ax0.plot(merged_data['date'], merged_data['polarity'].notnull().astype(int), 'go', alpha=0.3, label='Sentiment Data Available')
    ax0.set_title(f'Data Availability Overview ({date_range})')
    ax0.set_ylabel('Available (1=Yes)')
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(['No', 'Yes'])
    ax0.legend()
    
    # Price subplot
    ax1 = plt.subplot(3, 1, 2)
    ax1.plot(valid_data['date'], valid_data['close'], color='blue', linewidth=2)
    
    # Add volume as area plot with reduced opacity
    ax1v = ax1.twinx()
    ax1v.fill_between(valid_data['date'], 0, valid_data['volume'], color='blue', alpha=0.1)
    ax1v.set_ylabel('Trading Volume', color='blue')  # Removed alpha parameter
    ax1v.tick_params(axis='y', labelcolor='blue')  # Removed alpha parameter
    ax1v.grid(False)  # No grid on volume
    
    # Add price grid and labels
    ax1.set_ylabel('BTC Price (USD)', color='blue')
    ax1.set_title('Bitcoin Price with Trading Volume')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis to show less tick labels to avoid crowding
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Sentiment subplot
    ax2 = plt.subplot(3, 1, 3, sharex=ax1)
    
    # Plot polarity
    ax2.plot(valid_data['date'], valid_data['polarity'], color='green', linewidth=2, label='Polarity')
    
    # Plot subjectivity with lighter color
    ax2.plot(valid_data['date'], valid_data['subjectivity'], color='orange', linewidth=1.5, alpha=0.7, label='Subjectivity')
    
    # Add average polarity line
    avg_polarity = valid_data['polarity'].mean()
    ax2.axhline(y=avg_polarity, color='green', linestyle='--', alpha=0.5, label=f'Avg Polarity: {avg_polarity:.3f}')
    
    # Add correlation annotation
    corr_val = valid_data['polarity'].corr(valid_data['price_change'])
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax2.text(0.02, 0.95, f'Corr with Price Change: {corr_val:.3f}', transform=ax2.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
    
    ax2.set_ylabel('Sentiment Metrics')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_title('News Sentiment Metrics')
    
    # Format x-axis to show less tick labels to avoid crowding
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('price_and_sentiment_over_time.png', dpi=300)
    
    # Save an additional plot focusing on a narrower timeframe if we have a lot of data
    if len(valid_data) > 90:  # If we have more than 90 days of data
        # Focus on the last 90 days of data
        recent_data = valid_data.iloc[-90:]
        
        # Create a focused plot on this more recent data
        plt.figure(figsize=(14, 8))
        
        # Price subplot
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(recent_data['date'], recent_data['close'], color='blue', linewidth=2.5)
        ax1.set_ylabel('BTC Price (USD)', color='blue')
        ax1.set_title(f'Recent Bitcoin Price and News Sentiment (Last 90 Days)')
        ax1.grid(True, alpha=0.3)
        
        # Sentiment subplot
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(recent_data['date'], recent_data['polarity'], color='green', linewidth=2.5, label='Polarity')
        ax2.fill_between(recent_data['date'], 0, recent_data['polarity'], color='green', alpha=0.15)
        ax2.plot(recent_data['date'], recent_data['subjectivity'], color='orange', linewidth=2, label='Subjectivity')
        ax2.set_ylabel('Sentiment Metrics')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.setp([ax1.get_xticklabels(), ax2.get_xticklabels()], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('recent_price_and_sentiment.png', dpi=300)
    
    # Check if there's a lag effect (does news sentiment predict price changes?)
    for lag in [1, 2, 3]:
        merged_data[f'polarity_lag_{lag}'] = merged_data['polarity'].shift(lag)
        lag_corr = merged_data['price_change'].corr(merged_data[f'polarity_lag_{lag}'])
        print(f"\nCorrelation between price change and sentiment {lag} day(s) before: {lag_corr:.4f}")

def analyze_news_subject_impact(btc_df, news_df):
    """
    Analyze how different news subjects impact price changes
    
    :param btc_df: DataFrame with Bitcoin price data
    :param news_df: DataFrame with news articles
    :return: None (plots and prints results)
    """
    # Add price change column to Bitcoin data
    btc_df['price_change'] = btc_df['close'].pct_change() * 100
    btc_df['price_change_direction'] = btc_df['price_change'].apply(lambda x: 'increase' if x > 0 else 'decrease')
    
    # Convert DateTimeIndex to date for merging
    btc_df['date'] = btc_df.index.date
    
    # Check if date is already a date object or still datetime
    if hasattr(news_df['date'], 'dt'):
        news_df['date'] = news_df['date'].dt.date
    
    # Create a cross tabulation of news subjects per date
    subject_counts = pd.crosstab(news_df['date'], news_df['subject'])
    
    # Merge with Bitcoin data
    merged_data = pd.merge(btc_df, subject_counts, on='date', how='left')
    merged_data.fillna(0, inplace=True)
    
    # Calculate the correlation between subject mentions and price changes
    subject_correlations = {}
    for subject in subject_counts.columns:
        subject_correlations[subject] = np.corrcoef(merged_data['price_change'].values, 
                                                   merged_data[subject].values)[0, 1]
    
    # Sort and print correlations
    sorted_correlations = {k: v for k, v in sorted(subject_correlations.items(), 
                                                 key=lambda item: abs(item[1]), 
                                                 reverse=True)}
    print("\nCorrelation between news subject mentions and price changes:")
    for subject, corr in sorted_correlations.items():
        print(f"{subject}: {corr:.4f}")
    
    # Plot top 5 subjects by correlation strength
    top_subjects = list(sorted_correlations.keys())[:5]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_subjects, [sorted_correlations[s] for s in top_subjects])
    
    # Color positive correlations blue and negative correlations red
    for i, bar in enumerate(bars):
        if sorted_correlations[top_subjects[i]] < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    plt.title('Top 5 News Subjects by Correlation with Price Changes')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('subject_correlations.png')

def analyze_source_reliability(btc_df, news_df):
    """
    Analyze which news sources are most reliable indicators of price movement
    
    :param btc_df: DataFrame with Bitcoin price data
    :param news_df: DataFrame with news articles
    :return: None (plots and prints results)
    """
    # Add price change column to Bitcoin data
    btc_df['price_change'] = btc_df['close'].pct_change() * 100
    btc_df['next_day_price_change'] = btc_df['price_change'].shift(-1)
    
    # Convert DateTimeIndex to date for merging
    btc_df['date'] = btc_df.index.date
    
    # Check if date is already a date object or still datetime
    if hasattr(news_df['date'], 'dt'):
        news_df['date'] = news_df['date'].dt.date
    
    # Group by date and source, calculate average sentiment
    source_sentiment = news_df.groupby(['date', 'source'])['polarity'].mean().reset_index()
    
    # Merge with Bitcoin data
    merged_data = pd.merge(source_sentiment, btc_df[['date', 'price_change', 'next_day_price_change']], 
                          on='date', how='left')
    
    # Calculate the correlation between source sentiment and next day price change
    source_predictive_power = merged_data.groupby('source').apply(
        lambda x: np.corrcoef(x['polarity'].values, x['next_day_price_change'].values)[0, 1] 
        if len(x) > 5 else np.nan  # Only consider sources with enough data points
    ).dropna()
    
    # Sort and print correlations
    sorted_sources = source_predictive_power.sort_values(ascending=False)
    print("\nCorrelation between news source sentiment and next day price changes:")
    print(sorted_sources)
    
    # Plot source reliability
    plt.figure(figsize=(12, 6))
    bars = sorted_sources.plot(kind='bar')
    plt.title('News Source Reliability for Predicting Next-Day Price Movement')
    plt.ylabel('Correlation Coefficient (higher = more predictive)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig('source_reliability.png')

def analyze_price_direction_vs_polarity(btc_df, news_df):
    """
    Create a plot showing the relationship between price direction (positive/negative) and sentiment polarity over time.
    
    :param btc_df: DataFrame with Bitcoin price data
    :param news_df: DataFrame with news articles
    :return: None (plots and prints results)
    """
    # Add price change column to Bitcoin data
    btc_df['price_change'] = btc_df['close'].pct_change() * 100
    btc_df['price_direction'] = btc_df['price_change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Convert the date index of btc_df to a date object
    btc_df['date'] = btc_df.index.date
    
    # Make sure news_df date is a date object
    if hasattr(news_df['date'], 'dt'):
        news_df['date'] = news_df['date'].dt.date
    
    # Group news articles by date and calculate average sentiment metrics
    daily_news = news_df.groupby('date').agg({
        'polarity': 'mean',
    }).reset_index()
    
    # Merge price data with sentiment data
    merged_data = pd.merge(btc_df, daily_news, on='date', how='left')
    
    # Filter data to only include days where we have both sentiment and price data
    valid_data = merged_data.dropna(subset=['polarity', 'price_direction'])
    
    # Sort by date
    valid_data = valid_data.sort_values('date')
    
    # Create a figure with price direction and polarity
    plt.figure(figsize=(16, 10))
    
    # Plot Price Direction (as colored bars)
    ax1 = plt.subplot(3, 1, 1)
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in valid_data['price_direction']]
    ax1.bar(valid_data['date'], valid_data['price_change'], color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Price Change (%)')
    ax1.set_title('Bitcoin Price Change Direction and News Sentiment Polarity Over Time')
    
    # Add a legend for price direction
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Price Increase'),
        Patch(facecolor='red', alpha=0.7, label='Price Decrease')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot Price Direction as binary (1 for up, -1 for down)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in valid_data['price_direction']]
    ax2.bar(valid_data['date'], valid_data['price_direction'], color=colors, alpha=0.7)
    ax2.set_ylabel('Price Direction\n(1=Up, -1=Down)')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Down', 'Unchanged', 'Up'])
    
    # Plot Sentiment Polarity
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(valid_data['date'], valid_data['polarity'], color='blue', linewidth=1.5)
    ax3.fill_between(valid_data['date'], 0, valid_data['polarity'], 
                    color='blue', alpha=0.2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    mean_polarity = valid_data['polarity'].mean()
    ax3.axhline(y=mean_polarity, color='blue', linestyle='--', 
               linewidth=1, label=f'Mean Polarity: {mean_polarity:.3f}')
    ax3.set_ylabel('Sentiment Polarity')
    ax3.set_xlabel('Date')
    ax3.legend()
    
    # Calculate hit rate (% of times polarity correctly predicts price direction)
    valid_data['polarity_direction'] = valid_data['polarity'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    valid_data['match'] = (valid_data['polarity_direction'] == valid_data['price_direction']).astype(int)
    hit_rate = valid_data['match'].mean() * 100
    
    # Calculate correlations for different lag periods
    correlations = {}
    for lag in range(-5, 6):  # -5 to +5 days
        if lag < 0:
            # Sentiment lagging price (does price predict sentiment?)
            shifted_polarity = valid_data['polarity'].shift(-lag)
            corr = valid_data['price_direction'].corr(shifted_polarity)
            lag_type = f"Price leads Sentiment by {abs(lag)} day(s)"
        elif lag > 0:
            # Price lagging sentiment (does sentiment predict price?)
            shifted_price = valid_data['price_direction'].shift(lag)
            corr = valid_data['polarity'].corr(shifted_price)
            lag_type = f"Sentiment leads Price by {lag} day(s)"
        else:
            # Same day
            corr = valid_data['polarity'].corr(valid_data['price_direction'])
            lag_type = "Same day"
        
        correlations[lag_type] = corr
    
    # Add a text box with hit rate and correlation info
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    info_text = (f"Hit Rate (Polarity Direction = Price Direction): {hit_rate:.2f}%\n\n"
                f"Correlations:\n"
                f"Same day: {correlations['Same day']:.3f}\n"
                f"Sentiment leads Price by 1 day: {correlations['Sentiment leads Price by 1 day(s)']:.3f}\n"
                f"Price leads Sentiment by 1 day: {correlations['Price leads Sentiment by 1 day(s)']:.3f}")
    
    ax3.text(0.02, 0.05, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Format x-axis to avoid overcrowding
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('price_direction_vs_polarity.png', dpi=300)
    
    # Create an additional plot showing the relationship for the most recent 90 days
    if len(valid_data) > 90:
        recent_data = valid_data.iloc[-90:]
        
        # Calculate recent lag correlations
        recent_lag_correlations = {}
        for lag in range(-5, 6):  # -5 to +5 days
            if lag < 0:
                # Sentiment lagging price (does price predict sentiment?)
                shifted_polarity = recent_data['polarity'].shift(-lag)
                corr = recent_data['price_direction'].corr(shifted_polarity)
                lag_type = f"Price leads Sentiment by {abs(lag)} day(s)"
            elif lag > 0:
                # Price lagging sentiment (does sentiment predict price?)
                shifted_price = recent_data['price_direction'].shift(lag)
                corr = recent_data['polarity'].corr(shifted_price)
                lag_type = f"Sentiment leads Price by {lag} day(s)"
            else:
                # Same day
                corr = recent_data['polarity'].corr(recent_data['price_direction'])
                lag_type = "Same day"
            
            recent_lag_correlations[lag_type] = corr
        
        # Find the lag with the strongest correlation
        strongest_lag = max(recent_lag_correlations.items(), key=lambda x: abs(x[1]))
        
        plt.figure(figsize=(14, 12))  # Increased height for the lag correlation plot
        
        # Plot Price Direction
        ax1 = plt.subplot(3, 1, 1)
        colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in recent_data['price_direction']]
        ax1.bar(recent_data['date'], recent_data['price_direction'], color=colors, alpha=0.7)
        ax1.set_ylabel('Price Direction\n(1=Up, -1=Down)')
        ax1.set_yticks([-1, 0, 1])
        ax1.set_yticklabels(['Down', 'Unchanged', 'Up'])
        ax1.set_title('Recent Bitcoin Price Direction vs News Sentiment (Last 90 Days)')
        
        # Plot Sentiment Polarity
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(recent_data['date'], recent_data['polarity'], color='blue', linewidth=2)
        ax2.fill_between(recent_data['date'], 0, recent_data['polarity'], color='blue', alpha=0.2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        recent_mean = recent_data['polarity'].mean()
        ax2.axhline(y=recent_mean, color='blue', linestyle='--', 
                  linewidth=1, label=f'Mean Polarity: {recent_mean:.3f}')
        ax2.set_ylabel('Sentiment Polarity')
        ax2.legend()
        
        # Plot lag correlation
        ax3 = plt.subplot(3, 1, 3)
        lags = list(range(-5, 6))
        lag_values = []
        for lag in lags:
            if lag < 0:
                key = f"Price leads Sentiment by {abs(lag)} day(s)"
            elif lag > 0:
                key = f"Sentiment leads Price by {lag} day(s)"
            else:
                key = "Same day"
            lag_values.append(recent_lag_correlations[key])
        
        bars = ax3.bar(lags, lag_values)
        
        # Color the bars based on whether it's price leading sentiment or sentiment leading price
        for i, bar in enumerate(bars):
            if lags[i] < 0:  # Price leads sentiment
                bar.set_color('red')
            elif lags[i] > 0:  # Sentiment leads price
                bar.set_color('green')
            else:  # Same day
                bar.set_color('blue')
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Lag (Negative = Price leads Sentiment, Positive = Sentiment leads Price)')
        ax3.set_ylabel('Correlation')
        ax3.set_title('Lag Correlation Analysis')
        ax3.set_xticks(lags)
        
        # Annotate the strongest correlation
        ax3.annotate(f"Strongest: {strongest_lag[0]}\nCorr: {strongest_lag[1]:.3f}",
                   xy=(lags[list(recent_lag_correlations.values()).index(strongest_lag[1])], strongest_lag[1]),
                   xytext=(0, 20 if strongest_lag[1] > 0 else -20),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        # Recent hit rate
        recent_hit_rate = recent_data['match'].mean() * 100
        recent_corr = recent_data['polarity'].corr(recent_data['price_direction'])
        
        # Add text with recent stats
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        recent_text = (f"Recent Hit Rate: {recent_hit_rate:.2f}%\n"
                     f"Recent Correlation: {recent_corr:.3f}\n"
                     f"Strongest Lag Correlation: {strongest_lag[0]} ({strongest_lag[1]:.3f})")
        ax2.text(0.02, 0.95, recent_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Format x-axis for the first two plots
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('recent_price_direction_vs_polarity.png', dpi=300)
        
        # Create a second supplementary plot to visualize lagged polarity vs price direction
        if strongest_lag[0] != "Same day" and abs(strongest_lag[1]) > 0.1:  # Only if there's a meaningful lag correlation
            plt.figure(figsize=(14, 6))
            
            # Extract lag value from the strongest lag
            if "Price leads Sentiment" in strongest_lag[0]:
                lag_val = -int(strongest_lag[0].split("by")[1].split("day")[0].strip())
                shifted_polarity = recent_data['polarity'].shift(-lag_val)
                title = f"Price Direction Leading Sentiment by {abs(lag_val)} Days (Correlation: {strongest_lag[1]:.3f})"
                shifted_series = shifted_polarity
                original_series = recent_data['price_direction']
            else:
                lag_val = int(strongest_lag[0].split("by")[1].split("day")[0].strip()) 
                shifted_price = recent_data['price_direction'].shift(lag_val)
                title = f"Sentiment Leading Price Direction by {lag_val} Days (Correlation: {strongest_lag[1]:.3f})"
                shifted_series = recent_data['polarity']
                original_series = shifted_price
            
            plt.plot(recent_data['date'], original_series, 'b-', label='Original Series', linewidth=2)
            plt.plot(recent_data['date'], shifted_series, 'r--', label='Shifted Series', linewidth=2)
            plt.legend()
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('recent_lag_analysis.png', dpi=300)
    
    # Print summary statistics
    print("\n--- Price Direction vs Sentiment Polarity Analysis ---")
    print(f"Hit Rate (Polarity direction matches price direction): {hit_rate:.2f}%")
    print("\nCorrelations between sentiment polarity and price direction:")
    for lag_type, corr in sorted(correlations.items()):
        print(f"  {lag_type}: {corr:.4f}")
    
    # Perform point-biserial correlation (special case of Pearson correlation when one variable is dichotomous)
    from scipy.stats import pointbiserialr
    pb_corr, p_value = pointbiserialr(valid_data['price_direction'], valid_data['polarity'])
    print(f"\nPoint-Biserial Correlation: {pb_corr:.4f} (p-value: {p_value:.4f})")

def main():
    # Load the Bitcoin OHLCV data
    btc_df = load_data('BTCUSDT', '1d')
    
    # Load the news data
    news_df = load_news_data()
    
    print(f"Bitcoin data shape: {btc_df.shape}")
    print(f"News data shape: {news_df.shape}")
    
    print("\nBitcoin data sample:")
    print(btc_df.head())
    
    print("\nNews data sample:")
    print(news_df.head())
    
    # Analyze relationship between price changes and news sentiment
    analyze_price_sentiment_relation(btc_df, news_df)
    
    # Analyze impact of different news subjects
    analyze_news_subject_impact(btc_df, news_df)
    
    # Analyze reliability of different news sources
    analyze_source_reliability(btc_df, news_df)
    
    # Analyze price direction vs polarity
    analyze_price_direction_vs_polarity(btc_df, news_df)

if __name__ == "__main__":
    main()