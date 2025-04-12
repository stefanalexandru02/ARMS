import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import json
import ast
from datetime import datetime
import os
import warnings

# Filter some common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for reproducibility
np.random.seed(42)

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# Create output directories
ensure_dir("results")
ensure_dir("results/sentiment_clusters")
ensure_dir("results/crypto_clusters")
ensure_dir("results/source_clusters")

def load_and_preprocess_data(file_path):
    """Load and preprocess the cryptocurrency news data"""
    print("Loading data from:", file_path)
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Function to safely parse JSON strings and handle empty data
    def safe_parse_json(json_str, default=None):
        if pd.isna(json_str) or json_str == '':
            return default
        
        try:
            # Handle both string representation of lists and actual JSON
            if isinstance(json_str, str):
                if json_str.startswith('[') and json_str.endswith(']'):
                    return ast.literal_eval(json_str)
                elif json_str == '[]':
                    return []
            return json_str
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing: {json_str}, Error: {e}")
            return default
    
    # Parse sentiment column
    df['sentiment_parsed'] = df['sentiment'].apply(
        lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else {})
    
    # Extract sentiment features
    df['sentiment_class'] = df['sentiment_parsed'].apply(
        lambda x: x.get('class', 'unknown') if isinstance(x, dict) else 'unknown')
    df['sentiment_polarity'] = df['sentiment_parsed'].apply(
        lambda x: float(x.get('polarity', 0)) if isinstance(x, dict) else 0)
    df['sentiment_subjectivity'] = df['sentiment_parsed'].apply(
        lambda x: float(x.get('subjectivity', 0)) if isinstance(x, dict) else 0)
    
    # Parse linguistics columns and handle missing values
    for col in ['title_verbs', 'title_adjectives', 'text_verbs', 'text_adjectives']:
        df[f'{col}_parsed'] = df[col].apply(lambda x: safe_parse_json(x, []))
        df[f'{col}_count'] = df[f'{col}_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df[f'{col}_text'] = df[f'{col}_parsed'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    # Create a feature for tonality presence
    tonality_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['title_tonality_score'] = df['title_tonality'].map(tonality_map).fillna(0)
    df['text_tonality_score'] = df['text_tonality'].map(tonality_map).fillna(0)
    
    # Simplify crypto subject categorization
    df['crypto_category'] = df['subject'].apply(categorize_crypto_subject)
    
    # Create a combined feature of all linguistic elements
    df['all_linguistic_features'] = df.apply(
        lambda row: ' '.join([
            str(row['title_verbs_text']), 
            str(row['title_adjectives_text']),
            str(row['text_verbs_text']),
            str(row['text_adjectives_text'])
        ]), axis=1)
    
    return df

def categorize_crypto_subject(subject):
    """Categorize subjects into main crypto categories"""
    subject = str(subject).lower()
    
    if 'bitcoin' in subject or 'btc' in subject:
        return 'Bitcoin'
    elif 'ethereum' in subject or 'eth' in subject:
        return 'Ethereum'
    elif 'ripple' in subject or 'xrp' in subject:
        return 'Ripple'
    elif 'litecoin' in subject or 'ltc' in subject:
        return 'Litecoin'
    elif 'binance' in subject or 'bnb' in subject:
        return 'Binance'
    elif any(coin in subject for coin in ['altcoin', 'alt', 'token']):
        return 'Altcoins'
    elif any(word in subject for word in ['blockchain', 'defi', 'nft', 'web3', 'dao']):
        return 'Blockchain Tech'
    elif any(word in subject for word in ['regulation', 'sec', 'law', 'government']):
        return 'Regulation'
    elif any(word in subject for word in ['exchange', 'trading', 'market']):
        return 'Markets & Exchanges'
    else:
        return 'Other Crypto'

def run_sentiment_clustering(df, n_clusters=3):
    """Run clustering based on sentiment features"""
    print("\n=== SENTIMENT-BASED CLUSTERING ===")
    
    # Extract sentiment features
    features = df[[
        'sentiment_polarity', 
        'sentiment_subjectivity',
        'title_tonality_score', 
        'text_tonality_score'
    ]].values
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['sentiment_cluster'] = kmeans.fit_predict(scaled_features)
    
    # Map clusters to sentiment labels based on average polarity
    cluster_polarities = df.groupby('sentiment_cluster')['sentiment_polarity'].mean().sort_values()
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    cluster_map = {cluster: sentiment_labels[i] for i, cluster in enumerate(cluster_polarities.index)}
    df['sentiment_cluster_label'] = df['sentiment_cluster'].map(cluster_map)
    
    # Analyze clusters
    analyze_sentiment_clusters(df)
    
    # Create visualizations
    create_sentiment_visualizations(df, kmeans, features)
    
    return df

def analyze_sentiment_clusters(df):
    """Analyze the sentiment clusters"""
    result = {}
    
    print("\nSentiment Cluster Analysis:")
    for cluster_name, group in df.groupby('sentiment_cluster_label'):
        avg_polarity = group['sentiment_polarity'].mean()
        avg_subjectivity = group['sentiment_subjectivity'].mean()
        dominant_title_tone = group['title_tonality'].value_counts().index[0]
        dominant_text_tone = group['text_tonality'].value_counts().index[0]
        
        result[cluster_name] = {
            'size': len(group),
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'dominant_title_tone': dominant_title_tone,
            'dominant_text_tone': dominant_text_tone
        }
        
        print(f"\n{cluster_name} Sentiment Cluster ({len(group)} articles):")
        print(f"  Average Polarity: {avg_polarity:.2f}")
        print(f"  Average Subjectivity: {avg_subjectivity:.2f}")
        print(f"  Dominant Title Tonality: {dominant_title_tone}")
        print(f"  Dominant Text Tonality: {dominant_text_tone}")
        
        # Most common crypto subjects in this sentiment cluster
        top_subjects = group['crypto_category'].value_counts().head(3)
        print("  Top Crypto Subjects:")
        for subject, count in top_subjects.items():
            print(f"    - {subject}: {count} articles ({count/len(group)*100:.1f}%)")
            
        # Most common sources in this sentiment cluster
        top_sources = group['source'].value_counts().head(3)
        print("  Top Sources:")
        for source, count in top_sources.items():
            print(f"    - {source}: {count} articles ({count/len(group)*100:.1f}%)")
    
    # Save detailed results
    with open('results/sentiment_clusters/analysis.txt', 'w') as f:
        f.write("=== SENTIMENT CLUSTER ANALYSIS ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for cluster_name, stats in result.items():
            f.write(f"\n--- {cluster_name.upper()} CLUSTER ---\n")
            f.write(f"Size: {stats['size']} articles\n")
            f.write(f"Average Polarity: {stats['avg_polarity']:.4f}\n")
            f.write(f"Average Subjectivity: {stats['avg_subjectivity']:.4f}\n")
            f.write(f"Dominant Title Tonality: {stats['dominant_title_tone']}\n")
            f.write(f"Dominant Text Tonality: {stats['dominant_text_tone']}\n")
            
            # Top subjects in this sentiment cluster
            f.write("\nTop Crypto Subjects:\n")
            for subject, count in df[df['sentiment_cluster_label']==cluster_name]['crypto_category'].value_counts().head(5).items():
                pct = count / stats['size'] * 100
                f.write(f"  - {subject}: {count} articles ({pct:.1f}%)\n")
            
            # Top sources in this sentiment cluster
            f.write("\nTop Sources:\n")
            for source, count in df[df['sentiment_cluster_label']==cluster_name]['source'].value_counts().head(5).items():
                pct = count / stats['size'] * 100
                f.write(f"  - {source}: {count} articles ({pct:.1f}%)\n")
                
            # Sample titles
            f.write("\nSample Titles:\n")
            for i, title in enumerate(df[df['sentiment_cluster_label']==cluster_name]['title'].sample(min(5, stats['size'])).values):
                f.write(f"  {i+1}. {title}\n")
                
            f.write("\n" + "-" * 50 + "\n")
            
    return result

def create_sentiment_visualizations(df, kmeans, features):
    """Create visualizations for sentiment clusters"""
    # 1. Scatter plot of polarity vs subjectivity colored by cluster
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=df, 
        x='sentiment_polarity', 
        y='sentiment_subjectivity', 
        hue='sentiment_cluster_label',
        palette='viridis',
        alpha=0.6,
        s=80
    )
    plt.title('Sentiment Clusters: Polarity vs. Subjectivity', fontsize=15)
    plt.xlabel('Polarity (Negative to Positive)', fontsize=12)
    plt.ylabel('Subjectivity (Objective to Subjective)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Sentiment Cluster')
    plt.tight_layout()
    plt.savefig('results/sentiment_clusters/polarity_subjectivity.png', dpi=300)
    plt.close()
    
    # 2. Distribution of sentiment clusters over time
    plt.figure(figsize=(15, 8))
    df_time = df.groupby([pd.Grouper(key='date', freq='M'), 'sentiment_cluster_label']).size().unstack().fillna(0)
    df_time.plot(
        kind='area', 
        stacked=True, 
        alpha=0.7,
        colormap='viridis',
        ax=plt.gca()
    )
    plt.title('Sentiment Clusters Over Time', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig('results/sentiment_clusters/sentiment_over_time.png', dpi=300)
    plt.close()
    
    # 3. Heatmap of sentiment clusters by crypto category
    plt.figure(figsize=(14, 10))
    crosstab = pd.crosstab(df['crypto_category'], df['sentiment_cluster_label'])
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    sns.heatmap(
        crosstab_pct, 
        annot=True, 
        fmt='.1f', 
        cmap='YlGnBu',
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Distribution of Sentiment Clusters by Crypto Category', fontsize=15)
    plt.xlabel('Sentiment Cluster', fontsize=12)
    plt.ylabel('Crypto Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/sentiment_clusters/sentiment_by_category.png', dpi=300)
    plt.close()
    
    # 4. Bar chart of sentiment metrics by cluster
    sentiment_stats = df.groupby('sentiment_cluster_label')[['sentiment_polarity', 'sentiment_subjectivity']].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_stats.plot(
        kind='bar',
        colormap='viridis',
        ax=ax
    )
    plt.title('Average Sentiment Metrics by Cluster', fontsize=15)
    plt.xlabel('Sentiment Cluster', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/sentiment_clusters/average_sentiment.png', dpi=300)
    plt.close()

def run_crypto_clustering(df, max_clusters=8):
    """Run clustering focused on cryptocurrency subjects"""
    print("\n=== CRYPTOCURRENCY SUBJECT-BASED CLUSTERING ===")
    
    # Extract relevant features for crypto categorization
    
    # Feature 1: TF-IDF of titles - titles often contain the crypto names
    tfidf_vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    title_features = tfidf_vectorizer.fit_transform(df['title'].fillna(''))
    
    # Feature 2: One-hot encoded crypto categories
    crypto_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    crypto_one_hot = crypto_encoder.fit_transform(df[['crypto_category']])
    
    # Feature 3: Include linguistic features related to technical terms
    nlp_features = TfidfVectorizer(max_features=100, stop_words='english').fit_transform(df['all_linguistic_features'])
    
    # Convert sparse matrices to dense for combination
    title_dense = title_features.toarray()
    nlp_dense = nlp_features.toarray()
    
    # Combine features
    combined_features = np.hstack((title_dense, crypto_one_hot, nlp_dense))
    
    # Reduce dimensionality for better clustering
    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced_features = svd.fit_transform(combined_features)
    print(f"Reduced cryptocurrency features from {combined_features.shape[1]} to 50 components")
    
    # Determine optimal number of clusters using silhouette score
    optimal_k = find_optimal_clusters(reduced_features, max_k=max_clusters, method='silhouette')
    print(f"Optimal number of cryptocurrency clusters: {optimal_k}")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['crypto_cluster'] = kmeans.fit_predict(reduced_features)
    
    # Create descriptive names for the crypto clusters based on dominant cryptocurrencies
    crypto_cluster_names = generate_crypto_cluster_names(df)
    df['crypto_cluster_label'] = df['crypto_cluster'].map(crypto_cluster_names)
    
    # Analyze the clusters
    analyze_crypto_clusters(df)
    
    # Create visualizations
    create_crypto_visualizations(df)
    
    return df

def find_optimal_clusters(features, max_k=10, method='silhouette'):
    """Find optimal number of clusters using the specified method"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        if method == 'silhouette':
            score = silhouette_score(features, labels)
        elif method == 'calinski':
            score = calinski_harabasz_score(features, labels)
        else:
            score = -kmeans.inertia_  # Negative inertia for elbow method (higher is better)
            
        scores.append((k, score))
        
    # For silhouette and calinski, higher is better
    best_k, best_score = max(scores, key=lambda x: x[1])
    return best_k

def generate_crypto_cluster_names(df):
    """Generate descriptive names for crypto clusters based on dominant content"""
    crypto_cluster_names = {}
    
    for cluster in df['crypto_cluster'].unique():
        cluster_df = df[df['crypto_cluster'] == cluster]
        
        # Find dominant crypto category
        top_categories = cluster_df['crypto_category'].value_counts()
        dominant_category = top_categories.index[0] if len(top_categories) > 0 else 'Mixed'
        second_category = top_categories.index[1] if len(top_categories) > 1 else None
        
        # Find dominant topics from title keywords
        title_words = ' '.join(cluster_df['title'].fillna('').str.lower().tolist())
        
        # Create a descriptive name
        if second_category and top_categories[second_category] > top_categories[dominant_category] * 0.6:
            name = f"{dominant_category} & {second_category}"
        else:
            name = dominant_category
        
        crypto_cluster_names[cluster] = name
    
    return crypto_cluster_names

def analyze_crypto_clusters(df):
    """Analyze the cryptocurrency clusters"""
    result = {}
    
    print("\nCrypto Cluster Analysis:")
    for cluster_label, group in df.groupby('crypto_cluster_label'):
        # Basic stats
        cluster_size = len(group)
        crypto_categories = group['crypto_category'].value_counts()
        
        # Sentiment distribution
        sentiment_counts = group['sentiment_class'].value_counts()
        avg_polarity = group['sentiment_polarity'].mean()
        
        # Sources
        sources = group['source'].value_counts().head(3)
        
        # Most frequent verbs and adjectives
        verbs = []
        for verbs_list in group['title_verbs_parsed']:
            if isinstance(verbs_list, list):
                verbs.extend(verbs_list)
        verb_counts = pd.Series(verbs).value_counts().head(5)
        
        result[cluster_label] = {
            'size': cluster_size,
            'categories': crypto_categories.to_dict(),
            'sentiment': sentiment_counts.to_dict(),
            'avg_polarity': avg_polarity,
            'sources': sources.to_dict(),
            'top_verbs': verb_counts.to_dict()
        }
        
        print(f"\n{cluster_label} Cluster ({cluster_size} articles):")
        print("  Crypto Categories Distribution:")
        for category, count in crypto_categories.head(3).items():
            print(f"    - {category}: {count} articles ({count/cluster_size*100:.1f}%)")
        
        print(f"  Average Sentiment Polarity: {avg_polarity:.2f}")
        print(f"  Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"    - {sentiment}: {count} articles ({count/cluster_size*100:.1f}%)")
            
        print("  Top Sources:")
        for source, count in sources.items():
            print(f"    - {source}: {count} articles ({count/cluster_size*100:.1f}%)")
    
    # Save detailed analysis
    with open('results/crypto_clusters/analysis.txt', 'w') as f:
        f.write("=== CRYPTOCURRENCY CLUSTER ANALYSIS ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for cluster_name, stats in result.items():
            f.write(f"\n--- {cluster_name.upper()} CLUSTER ---\n")
            f.write(f"Size: {stats['size']} articles\n")
            
            f.write("\nCrypto Categories:\n")
            for category, count in stats['categories'].items():
                pct = count / stats['size'] * 100
                f.write(f"  - {category}: {count} articles ({pct:.1f}%)\n")
            
            f.write(f"\nAverage Sentiment Polarity: {stats['avg_polarity']:.4f}\n")
            f.write("Sentiment Distribution:\n")
            for sentiment, count in stats['sentiment'].items():
                pct = count / stats['size'] * 100
                f.write(f"  - {sentiment}: {count} articles ({pct:.1f}%)\n")
            
            f.write("\nTop Sources:\n")
            for source, count in stats['sources'].items():
                pct = count / stats['size'] * 100
                f.write(f"  - {source}: {count} articles ({pct:.1f}%)\n")
            
            f.write("\nTop Verbs:\n")
            for verb, count in stats['top_verbs'].items():
                f.write(f"  - {verb}: {count} occurrences\n")
                
            # Sample titles
            f.write("\nSample Titles:\n")
            for i, title in enumerate(df[df['crypto_cluster_label']==cluster_name]['title'].sample(min(5, stats['size'])).values):
                f.write(f"  {i+1}. {title}\n")
                
            f.write("\n" + "-" * 50 + "\n")
    
    return result

def create_crypto_visualizations(df):
    """Create visualizations for cryptocurrency clusters"""
    # 1. Bar chart of cryptocurrency categories in each cluster
    plt.figure(figsize=(16, 10))
    
    # Create data for stacked bar chart
    crypto_cluster_data = pd.crosstab(df['crypto_cluster_label'], df['crypto_category'])
    
    # Plot stacked bar chart
    crypto_cluster_data.plot(
        kind='bar', 
        stacked=True,
        colormap='tab20',
        figsize=(16, 10)
    )
    
    plt.title('Distribution of Cryptocurrency Categories by Cluster', fontsize=15)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Cryptocurrency Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/crypto_clusters/category_distribution.png', dpi=300)
    plt.close()
    
    # 2. Heatmap of sentiment across crypto clusters
    plt.figure(figsize=(12, 8))
    sentiment_crypto_data = pd.crosstab(df['crypto_cluster_label'], df['sentiment_cluster_label'])
    sentiment_crypto_pct = sentiment_crypto_data.div(sentiment_crypto_data.sum(axis=1), axis=0) * 100
    
    sns.heatmap(
        sentiment_crypto_pct, 
        annot=True, 
        fmt='.1f', 
        cmap='coolwarm',
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Sentiment Distribution across Cryptocurrency Clusters', fontsize=15)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Cryptocurrency Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/crypto_clusters/sentiment_distribution.png', dpi=300)
    plt.close()
    
    # 3. Time series of crypto clusters
    plt.figure(figsize=(16, 8))
    df_time = df.groupby([pd.Grouper(key='date', freq='M'), 'crypto_cluster_label']).size().unstack().fillna(0)
    
    df_time.plot(
        kind='line',
        marker='o',
        markersize=5,
        linewidth=2,
        alpha=0.8,
        colormap='tab20',
        ax=plt.gca()
    )
    
    plt.title('Cryptocurrency Clusters Over Time', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Crypto Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/crypto_clusters/clusters_over_time.png', dpi=300)
    plt.close()
    
    # 4. Pie chart of overall cluster distribution
    plt.figure(figsize=(12, 12))
    df['crypto_cluster_label'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(df['crypto_cluster_label'].unique()),
        colormap='tab20',
        textprops={'fontsize': 12}
    )
    plt.title('Distribution of Articles across Cryptocurrency Clusters', fontsize=15)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('results/crypto_clusters/cluster_distribution_pie.png', dpi=300)
    plt.close()

def run_source_clustering(df, max_clusters=10):
    """Run clustering based on news sources"""
    print("\n=== NEWS SOURCE-BASED CLUSTERING ===")
    
    # Extract source information
    sources = df['source'].fillna('unknown').values
    
    # Create one-hot encoding for sources
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    source_features = encoder.fit_transform(sources.reshape(-1, 1))
    
    # Enhance source features with their typical content
    # Extract linguistic features associated with each source
    count_vectorizer = CountVectorizer(max_features=200, stop_words='english')
    text_features = count_vectorizer.fit_transform(df['all_linguistic_features'])
    
    # Combine features
    combined_features = np.hstack((source_features, text_features.toarray()))
    
    # Reduce dimensionality
    svd = TruncatedSVD(n_components=min(50, combined_features.shape[1]-1), random_state=42)
    reduced_features = svd.fit_transform(combined_features)
    
    # Determine optimal number of clusters
    optimal_k = find_optimal_clusters(reduced_features, max_k=max_clusters)
    print(f"Optimal number of source clusters: {optimal_k}")
    
    # Apply clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['source_cluster'] = kmeans.fit_predict(reduced_features)
    
    # Generate descriptive names for source clusters
    source_cluster_names = generate_source_cluster_names(df)
    df['source_cluster_label'] = df['source_cluster'].map(source_cluster_names)
    
    # Analyze the clusters
    analyze_source_clusters(df)
    
    # Create visualizations
    create_source_visualizations(df)
    
    return df

def generate_source_cluster_names(df):
    """Generate descriptive names for source clusters"""
    source_cluster_names = {}
    
    for cluster in df['source_cluster'].unique():
        cluster_df = df[df['source_cluster'] == cluster]
        
        # Find top sources in this cluster
        top_sources = cluster_df['source'].value_counts().head(3)
        
        if len(top_sources) == 0:
            source_cluster_names[cluster] = f"Source Cluster {cluster}"
            continue
            
        # Create descriptor based on top sources
        if len(top_sources) == 1:
            name = top_sources.index[0]
        else:
            first_source = top_sources.index[0]
            
            # Check if one source is dominant
            if top_sources.iloc[0] > top_sources.sum() * 0.7:
                name = f"{first_source} Group"
            else:
                # Group by type of sources
                mainstream_media = ['reuters', 'bloomberg', 'cnbc', 'forbes', 'wsj', 'cnn', 'bbc']
                crypto_media = ['coindesk', 'cointelegraph', 'bitcoin.com', 'crypto news', 'the block']
                
                top_source_list = [s.lower() for s in top_sources.index]
                
                if any(source in mainstream_media for source in top_source_list):
                    if all(source in mainstream_media for source in top_source_list):
                        name = "Mainstream Media"
                    else:
                        name = "Mixed Media"
                elif all(source in crypto_media for source in top_source_list):
                    name = "Crypto Media"
                else:
                    # Just use the top two sources
                    name = f"{top_sources.index[0]} & {top_sources.index[1]}"
        
        source_cluster_names[cluster] = name
    
    return source_cluster_names

def analyze_source_clusters(df):
    """Analyze the news source clusters"""
    result = {}
    
    print("\nSource Cluster Analysis:")
    for cluster_label, group in df.groupby('source_cluster_label'):
        # Basic stats
        cluster_size = len(group)
        sources = group['source'].value_counts().head(5)
        
        # Sentiment analysis
        avg_polarity = group['sentiment_polarity'].mean()
        avg_subjectivity = group['sentiment_subjectivity'].mean()
        
        # Content focus
        crypto_focus = group['crypto_category'].value_counts().head(3)
        
        result[cluster_label] = {
            'size': cluster_size,
            'sources': sources.to_dict(),
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'crypto_focus': crypto_focus.to_dict()
        }
        
        print(f"\n{cluster_label} Cluster ({cluster_size} articles):")
        print("  Top Sources:")
        for source, count in sources.items():
            print(f"    - {source}: {count} articles ({count/cluster_size*100:.1f}%)")
        
        print(f"  Average Sentiment: Polarity={avg_polarity:.2f}, Subjectivity={avg_subjectivity:.2f}")
        print("  Crypto Focus:")
        for category, count in crypto_focus.items():
            print(f"    - {category}: {count} articles ({count/cluster_size*100:.1f}%)")
    
    # Save detailed analysis
    with open('results/source_clusters/analysis.txt', 'w') as f:
        f.write("=== NEWS SOURCE CLUSTER ANALYSIS ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for cluster_name, stats in result.items():
            f.write(f"\n--- {cluster_name.upper()} CLUSTER ---\n")
            f.write(f"Size: {stats['size']} articles\n")
            
            f.write("\nTop Sources:\n")
            for source, count in stats['sources'].items():
                pct = count / stats['size'] * 100
                f.write(f"  - {source}: {count} articles ({pct:.1f}%)\n")
            
            f.write(f"\nAverage Sentiment: Polarity={stats['avg_polarity']:.4f}, Subjectivity={stats['avg_subjectivity']:.4f}\n")
            
            f.write("\nCrypto Focus:\n")
            for category, count in stats['crypto_focus'].items():
                pct = count / stats['size'] * 100
                f.write(f"  - {category}: {count} articles ({pct:.1f}%)\n")
                
            # Sample titles
            f.write("\nSample Titles:\n")
            for i, title in enumerate(df[df['source_cluster_label']==cluster_name]['title'].sample(min(5, stats['size'])).values):
                f.write(f"  {i+1}. {title}\n")
                
            f.write("\n" + "-" * 50 + "\n")
    
    return result

def create_source_visualizations(df):
    """Create visualizations for news source clusters"""
    # 1. Stacked bar chart of sources in each cluster
    plt.figure(figsize=(16, 10))
    
    # Get top 10 sources for readability
    top_sources = df['source'].value_counts().head(10).index
    filtered_df = df[df['source'].isin(top_sources)].copy()
    
    # Create data for stacked bar chart
    source_data = pd.crosstab(filtered_df['source_cluster_label'], filtered_df['source'])
    
    # Plot stacked bar chart
    source_data.plot(
        kind='bar', 
        stacked=True,
        colormap='tab20',
        figsize=(16, 10)
    )
    
    plt.title('Distribution of News Sources by Cluster', fontsize=15)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='News Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/source_clusters/source_distribution.png', dpi=300)
    plt.close()
    
    # 2. Heatmap of sentiment metrics by source cluster
    plt.figure(figsize=(10, 6))
    sentiment_data = df.groupby('source_cluster_label')[['sentiment_polarity', 'sentiment_subjectivity']].mean().sort_values('sentiment_polarity')
    
    sns.heatmap(
        sentiment_data, 
        annot=True, 
        fmt='.2f', 
        cmap='RdYlGn',
        cbar_kws={'label': 'Score'}
    )
    plt.title('Average Sentiment Metrics by Source Cluster', fontsize=15)
    plt.tight_layout()
    plt.savefig('results/source_clusters/sentiment_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Bubble chart of source clusters by size, sentiment, and subjectivity
    plt.figure(figsize=(14, 10))
    
    # Prepare data
    cluster_stats = df.groupby('source_cluster_label').agg({
        'sentiment_polarity': 'mean',
        'sentiment_subjectivity': 'mean',
        'source_cluster': 'size'
    }).rename(columns={'source_cluster': 'size'})
    
    # Create bubble chart
    sns.scatterplot(
        data=cluster_stats,
        x='sentiment_polarity',
        y='sentiment_subjectivity',
        size='size',
        sizes=(100, 2000),
        alpha=0.6,
        palette='viridis',
        legend='brief'
    )
    
    # Add labels
    for idx, row in cluster_stats.iterrows():
        plt.annotate(
            idx,
            (row['sentiment_polarity'], row['sentiment_subjectivity']),
            ha='center',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.title('Source Clusters: Sentiment Polarity vs Subjectivity', fontsize=15)
    plt.xlabel('Sentiment Polarity (Negative to Positive)', fontsize=12)
    plt.ylabel('Subjectivity (Objective to Subjective)', fontsize=12)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/source_clusters/sentiment_bubble.png', dpi=300)
    plt.close()
    
    # 4. Heatmap showing crypto focus by source cluster
    plt.figure(figsize=(14, 10))
    crypto_focus = pd.crosstab(df['source_cluster_label'], df['crypto_category'])
    crypto_focus_pct = crypto_focus.div(crypto_focus.sum(axis=1), axis=0) * 100
    
    sns.heatmap(
        crypto_focus_pct, 
        annot=True, 
        fmt='.1f', 
        cmap='YlGnBu',
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Cryptocurrency Focus by Source Cluster', fontsize=15)
    plt.xlabel('Cryptocurrency Category', fontsize=12)
    plt.ylabel('Source Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/source_clusters/crypto_focus.png', dpi=300)
    plt.close()

def combine_cluster_information(df):
    """Combine insights from all three clustering approaches"""
    # Create a combined report showing relationships between the three analyses
    print("\n=== COMBINED CLUSTERING ANALYSIS ===")
    
    # 1. Create contingency tables
    sentiment_crypto = pd.crosstab(df['sentiment_cluster_label'], df['crypto_cluster_label'])
    sentiment_source = pd.crosstab(df['sentiment_cluster_label'], df['source_cluster_label'])
    crypto_source = pd.crosstab(df['crypto_cluster_label'], df['source_cluster_label'])
    
    # 2. Calculate percentages
    sentiment_crypto_pct = sentiment_crypto.div(sentiment_crypto.sum(axis=1), axis=0) * 100
    sentiment_source_pct = sentiment_source.div(sentiment_source.sum(axis=1), axis=0) * 100
    crypto_source_pct = crypto_source.div(crypto_source.sum(axis=1), axis=0) * 100
    
    # Create visualizations
    
    # 1. Heatmap: Sentiment clusters vs. Crypto clusters
    plt.figure(figsize=(16, 8))
    sns.heatmap(
        sentiment_crypto_pct,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Distribution of Crypto Clusters within each Sentiment Cluster', fontsize=15)
    plt.xlabel('Cryptocurrency Cluster', fontsize=12)
    plt.ylabel('Sentiment Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/combined_sentiment_crypto.png', dpi=300)
    plt.close()
    
    # 2. Heatmap: Sentiment clusters vs. Source clusters
    plt.figure(figsize=(16, 8))
    sns.heatmap(
        sentiment_source_pct,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Distribution of Source Clusters within each Sentiment Cluster', fontsize=15)
    plt.xlabel('Source Cluster', fontsize=12)
    plt.ylabel('Sentiment Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/combined_sentiment_source.png', dpi=300)
    plt.close()
    
    # 3. Heatmap: Crypto clusters vs. Source clusters
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        crypto_source_pct,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Distribution of Source Clusters within each Crypto Cluster', fontsize=15)
    plt.xlabel('Source Cluster', fontsize=12)
    plt.ylabel('Cryptocurrency Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/combined_crypto_source.png', dpi=300)
    plt.close()
    
    # Save combined results
    df.to_csv('results/combined_clustering_results.csv', index=False)
    
    return {
        'sentiment_crypto': sentiment_crypto_pct,
        'sentiment_source': sentiment_source_pct,
        'crypto_source': crypto_source_pct
    }

def generate_summary_report(df, combined_analysis):
    """Generate a comprehensive summary report of all clustering analyses"""
    with open('results/clustering_summary_report.txt', 'w') as f:
        f.write("=== CRYPTOCURRENCY NEWS CLUSTERING ANALYSIS SUMMARY ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total articles analyzed: {len(df)}\n\n")
        
        # 1. Sentiment Clustering Summary
        f.write("=== SENTIMENT CLUSTERING SUMMARY ===\n\n")
        
        sentiment_counts = df['sentiment_cluster_label'].value_counts()
        for sentiment, count in sentiment_counts.items():
            f.write(f"{sentiment} Cluster: {count} articles ({count/len(df)*100:.1f}%)\n")
            
        f.write("\nKey insights from sentiment analysis:\n")
        f.write("- Articles with positive sentiment tend to focus on: ")
        pos_focus = df[df['sentiment_cluster_label']=='Positive']['crypto_category'].value_counts().head(2).index
        f.write(", ".join(pos_focus) + "\n")
        
        f.write("- Articles with negative sentiment tend to focus on: ")
        neg_focus = df[df['sentiment_cluster_label']=='Negative']['crypto_category'].value_counts().head(2).index
        f.write(", ".join(neg_focus) + "\n")
        
        # 2. Cryptocurrency Clustering Summary
        f.write("\n\n=== CRYPTOCURRENCY CLUSTERING SUMMARY ===\n\n")
        
        crypto_counts = df['crypto_cluster_label'].value_counts()
        for crypto, count in crypto_counts.items():
            f.write(f"{crypto} Cluster: {count} articles ({count/len(df)*100:.1f}%)\n")
            
        f.write("\nKey insights from cryptocurrency analysis:\n")
        f.write("- The largest cryptocurrency clusters are focused on: ")
        largest_crypto = crypto_counts.index[0]
        f.write(f"{largest_crypto}\n")
        
        # 3. Source Clustering Summary
        f.write("\n\n=== NEWS SOURCE CLUSTERING SUMMARY ===\n\n")
        
        source_counts = df['source_cluster_label'].value_counts()
        for source, count in source_counts.items():
            f.write(f"{source} Cluster: {count} articles ({count/len(df)*100:.1f}%)\n")
            
        f.write("\nKey insights from source analysis:\n")
        f.write("- Different source clusters show varying sentiment biases\n")
        
        # 4. Combined Analysis Insights
        f.write("\n\n=== COMBINED ANALYSIS INSIGHTS ===\n\n")
        
        # Find most positive crypto cluster
        crypto_sentiment = df.groupby('crypto_cluster_label')['sentiment_polarity'].mean().sort_values(ascending=False)
        most_positive_crypto = crypto_sentiment.index[0]
        most_negative_crypto = crypto_sentiment.index[-1]
        
        f.write(f"- Most positive coverage: {most_positive_crypto} (avg. polarity: {crypto_sentiment.iloc[0]:.2f})\n")
        f.write(f"- Most negative coverage: {most_negative_crypto} (avg. polarity: {crypto_sentiment.iloc[-1]:.2f})\n")
        
        # Find source with most positive/negative coverage
        source_sentiment = df.groupby('source_cluster_label')['sentiment_polarity'].mean().sort_values(ascending=False)
        most_positive_source = source_sentiment.index[0]
        most_negative_source = source_sentiment.index[-1]
        
        f.write(f"- Most positive sources: {most_positive_source} (avg. polarity: {source_sentiment.iloc[0]:.2f})\n")
        f.write(f"- Most negative sources: {most_negative_source} (avg. polarity: {source_sentiment.iloc[-1]:.2f})\n")
        
        # Show how different sources cover different cryptos
        f.write("\nSource preferences for cryptocurrency topics:\n")
        for source in source_counts.index[:3]:  # Top 3 source clusters
            top_crypto = df[df['source_cluster_label']==source]['crypto_category'].value_counts().head(1)
            f.write(f"- {source} focuses on: {top_crypto.index[0]} ({top_crypto.iloc[0]} articles)\n")
        
        # Conclusions
        f.write("\n\n=== CONCLUSIONS ===\n\n")
        f.write("1. The sentiment analysis reveals distinct clusters in how cryptocurrency news is framed,\n")
        f.write("   with clear differences in the emotions conveyed across articles.\n\n")
        
        f.write("2. Cryptocurrency clustering shows that coverage tends to group around major coins\n")
        f.write("   and blockchain concepts rather than being evenly distributed.\n\n")
        
        f.write("3. The source analysis demonstrates that different news outlets have recognizable\n")
        f.write("   patterns in their cryptocurrency coverage, with some focusing on specific coins\n")
        f.write("   or displaying particular sentiment biases.\n\n")
        
        f.write("4. The combined analysis reveals important relationships between the source of news,\n")
        f.write("   the cryptocurrency being covered, and the sentiment of the coverage.\n")
        
        print("\nSummary report generated: results/clustering_summary_report.txt")

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('/Users/stefanalexandru/Desktop/ARMS/crypto_nlp_results_intermediate.csv')
    
    # Run sentiment clustering (3 clusters as requested)
    df = run_sentiment_clustering(df, n_clusters=3)
    
    # Run cryptocurrency subject clustering
    df = run_crypto_clustering(df)
    
    # Run news source clustering
    df = run_source_clustering(df)
    
    # Combine all clustering information
    combined_analysis = combine_cluster_information(df)
    
    # Generate summary report
    generate_summary_report(df, combined_analysis)
    
    print("\nAll clustering analyses completed successfully!")
    print("Results are saved in the 'results' directory with separate folders for each analysis type.")

if __name__ == "__main__":
    main()