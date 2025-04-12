import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import ast
from datetime import datetime
import re

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """Load and preprocess the cryptocurrency news data with NLP analysis"""
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
    
    # Create a combined feature of all linguistic elements
    df['all_linguistic_features'] = df.apply(
        lambda row: ' '.join([
            str(row['title_verbs_text']), 
            str(row['title_adjectives_text']),
            str(row['text_verbs_text']),
            str(row['text_adjectives_text'])
        ]), axis=1)
    
    return df

def create_feature_vectors(df):
    """Create feature vectors for clustering from linguistic data"""
    # Create TF-IDF vectors for linguistic text
    tfidf = TfidfVectorizer(
        max_features=100, 
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Apply TF-IDF to the combined linguistic features
    linguistic_features = tfidf.fit_transform(df['all_linguistic_features'].fillna(''))
    print(f"Generated {linguistic_features.shape[1]} linguistic TF-IDF features")
    
    # Create a DataFrame with numerical features
    numerical_features = df[[
        'sentiment_polarity', 'sentiment_subjectivity',
        'title_verbs_count', 'title_adjectives_count',
        'text_verbs_count', 'text_adjectives_count',
        'title_tonality_score', 'text_tonality_score'
    ]].fillna(0)
    
    # Standardize numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    
    # Convert sparse matrix to dense for combining with other features
    linguistic_dense = linguistic_features.toarray()
    
    # Combine all features
    combined_features = np.hstack((linguistic_dense, scaled_numerical))
    print(f"Generated feature matrix with shape: {combined_features.shape}")
    
    return combined_features, tfidf.get_feature_names_out()

def perform_clustering(features, n_clusters=5):
    """Perform K-means clustering on the feature vectors"""
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    # Also try DBSCAN for comparison
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(features)
    
    return clusters, dbscan_clusters, kmeans

def generate_cluster_names(cluster_stats):
    """Generate meaningful names for clusters based on their characteristics"""
    cluster_names = {}
    
    for cluster_id, stats in cluster_stats.items():
        # Get dominant sentiment
        sentiment_counts = stats['sentiment_distribution']
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        # Get dominant subject
        subject_counts = stats['common_subjects']
        dominant_subject = list(subject_counts.keys())[0] if subject_counts else 'general'
        
        # Get dominant source
        source_counts = stats['common_sources']
        dominant_source = list(source_counts.keys())[0] if source_counts else 'various'
        
        # Get average sentiment metrics
        polarity = stats['avg_polarity']
        subjectivity = stats['avg_subjectivity']
        
        # Get dominant tonality
        title_tonality = stats['title_tonality']
        dominant_title_tone = max(title_tonality.items(), key=lambda x: x[1])[0] if title_tonality else 'neutral'
        
        # Create descriptive name components
        sentiment_desc = dominant_sentiment
        if polarity > 0.2:
            sentiment_desc = "Strongly Positive"
        elif polarity > 0:
            sentiment_desc = "Positive"
        elif polarity < -0.2:
            sentiment_desc = "Strongly Negative"
        elif polarity < 0:
            sentiment_desc = "Negative"
        else:
            sentiment_desc = "Neutral"
        
        subject_desc = dominant_subject.capitalize()
        
        # Create descriptive name using format: "Sentiment Subject (Source)"
        descriptive_name = f"{sentiment_desc} {subject_desc} ({dominant_source})"
        
        # For special cases with very distinct characteristics
        if subjectivity > 0.5:
            descriptive_name = "Subjective " + descriptive_name
        elif subjectivity < 0.2:
            descriptive_name = "Objective " + descriptive_name
            
        # Store the mapping between cluster_id and descriptive name
        cluster_names[cluster_id] = descriptive_name
    
    return cluster_names

def visualize_clusters(features, clusters, title, df=None):
    """Create a 2D visualization of clusters using PCA with enhanced interpretability"""
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    # Create a DataFrame for easier plotting
    plot_df = pd.DataFrame({
        'PC1': reduced_features[:, 0],
        'PC2': reduced_features[:, 1],
        'Cluster': clusters
    })
    
    # Add informative features if df is provided
    if df is not None:
        # Add relevant columns for better interpretation
        plot_df['Subject'] = df['subject'].values
        plot_df['Sentiment'] = df['sentiment_class'].values
        plot_df['Title'] = df['title'].values
        plot_df['Source'] = df['source'].values
    
    # Get unique clusters for coloring
    unique_clusters = sorted(plot_df['Cluster'].unique())
    num_clusters = len(unique_clusters)
    
    # If this is K-means, try to generate descriptive names
    if title == "K-means Clusters" and df is not None:
        # Get temporary cluster stats for naming
        temp_df = df.copy()
        temp_df['cluster'] = clusters
        temp_stats = {}
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_df = temp_df[temp_df['cluster'] == cluster_id]
            
            # Basic stats for naming
            sentiment_counts = cluster_df['sentiment_class'].value_counts().to_dict()
            subjects = cluster_df['subject'].value_counts().head(3).to_dict()
            sources = cluster_df['source'].value_counts().head(3).to_dict()
            avg_polarity = cluster_df['sentiment_polarity'].mean()
            avg_subjectivity = cluster_df['sentiment_subjectivity'].mean()
            title_tonality = cluster_df['title_tonality'].value_counts().to_dict()
            
            temp_stats[cluster_id] = {
                'sentiment_distribution': sentiment_counts,
                'common_subjects': subjects,
                'common_sources': sources,
                'avg_polarity': avg_polarity,
                'avg_subjectivity': avg_subjectivity,
                'title_tonality': title_tonality
            }
        
        # Generate descriptive names
        cluster_labels = generate_cluster_names(temp_stats)
    else:
        # Map cluster numbers to default labels
        cluster_labels = {i: f"Cluster {i}" for i in unique_clusters if i != -1}
    
    # Handle DBSCAN noise points
    if -1 in unique_clusters:
        cluster_labels[-1] = "Noise"
        
    # Create custom colormap
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    
    # Create multiple subplots for better visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(f'{title} - PCA Visualization', fontsize=16)
    
    # 1. Basic cluster plot with improved labeling
    ax = axes[0, 0]
    for i, cluster in enumerate(unique_clusters):
        cluster_points = plot_df[plot_df['Cluster'] == cluster]
        ax.scatter(
            cluster_points['PC1'], 
            cluster_points['PC2'], 
            c=[colors[i]], 
            label=cluster_labels[cluster],
            alpha=0.7,
            s=50,
            edgecolors='w',
            linewidths=0.5
        )
    
    ax.set_title('Clusters with Descriptive Labels')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, alpha=0.3)
    ax.legend(title="Cluster Groups", loc='best', fontsize='small')
    
    # 2. Plot by subject if available
    ax = axes[0, 1]
    if df is not None:
        # Get top subjects for better visualization
        top_subjects = df['subject'].value_counts().index[:7].tolist()
        for subject in top_subjects:
            subject_points = plot_df[plot_df['Subject'] == subject]
            ax.scatter(
                subject_points['PC1'], 
                subject_points['PC2'], 
                label=subject,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidths=0.5
            )
        
        ax.set_title('Clusters by Subject')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid(True, alpha=0.3)
        ax.legend(title="Subjects", loc='best')
    
    # 3. Plot by sentiment if available
    ax = axes[1, 0]
    if df is not None:
        for sentiment in plot_df['Sentiment'].unique():
            sentiment_points = plot_df[plot_df['Sentiment'] == sentiment]
            ax.scatter(
                sentiment_points['PC1'], 
                sentiment_points['PC2'], 
                label=sentiment,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidths=0.5
            )
        
        ax.set_title('Clusters by Sentiment')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sentiment", loc='best')
    
    # 4. Plot by source if available
    ax = axes[1, 1]
    if df is not None:
        # Get top sources for better visualization
        top_sources = df['source'].value_counts().index[:7].tolist()
        for source in top_sources:
            source_points = plot_df[plot_df['Source'] == source]
            ax.scatter(
                source_points['PC1'], 
                source_points['PC2'], 
                label=source,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidths=0.5
            )
        
        ax.set_title('Clusters by Source')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sources", loc='best')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'crypto_news_clusters_{title.replace(" ", "_")}.png', dpi=300)
    plt.close()
    
    # Create dendrograms to visualize cluster hierarchy for K-means clustering
    if title == "K-means Clusters":
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        # Extract cluster centers from the K-means model
        # This is a workaround since we don't have the model here
        # We'll compute the mean of each feature for each cluster
        cluster_centers = np.zeros((len(unique_clusters), reduced_features.shape[1]))
        
        for i, cluster in enumerate(unique_clusters):
            cluster_points = reduced_features[clusters == cluster]
            if len(cluster_points) > 0:
                cluster_centers[i] = cluster_points.mean(axis=0)
        
        # Generate linkage matrix
        Z = linkage(cluster_centers, 'ward')
        
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=[f"Cluster {i}" for i in unique_clusters])
        plt.title('Hierarchical Clustering of K-means Cluster Centers')
        plt.xlabel('Clusters')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig('crypto_news_cluster_dendrogram.png', dpi=300)
        plt.close()
        
        # Also create a 3D visualization for better understanding
        from mpl_toolkits.mplot3d import Axes3D
        
        # If we have more than 2 PCA components
        if features.shape[1] >= 3:
            pca3d = PCA(n_components=3)
            reduced_features_3d = pca3d.fit_transform(features)
            
            plot_df_3d = pd.DataFrame({
                'PC1': reduced_features_3d[:, 0],
                'PC2': reduced_features_3d[:, 1],
                'PC3': reduced_features_3d[:, 2],
                'Cluster': clusters
            })
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, cluster in enumerate(unique_clusters):
                cluster_points = plot_df_3d[plot_df_3d['Cluster'] == cluster]
                ax.scatter(
                    cluster_points['PC1'], 
                    cluster_points['PC2'], 
                    cluster_points['PC3'],
                    c=[colors[i]], 
                    label=cluster_labels[cluster],
                    alpha=0.7,
                    s=50,
                    edgecolors='w',
                    linewidths=0.5
                )
            
            # Set labels and title
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            ax.set_title('3D Visualization of Clusters')
            
            # Add a legend
            ax.legend(title="Cluster Groups")
            
            # Save the 3D plot
            plt.tight_layout()
            plt.savefig(f'crypto_news_clusters_3D_{title.replace(" ", "_")}.png', dpi=300)
            plt.close()
    
    return reduced_features

def analyze_clusters(df, clusters, feature_names=None, top_n=10):
    """Analyze the characteristics of each cluster"""
    df['cluster'] = clusters
    cluster_stats = {}
    
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:  # Noise points in DBSCAN
            continue
            
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Calculate sentiment statistics
        sentiment_counts = cluster_df['sentiment_class'].value_counts()
        avg_polarity = cluster_df['sentiment_polarity'].mean()
        avg_subjectivity = cluster_df['sentiment_subjectivity'].mean()
        
        # Find most common subjects
        subjects = cluster_df['subject'].value_counts().head(3)
        
        # Find most common sources
        sources = cluster_df['source'].value_counts().head(3)
        
        # Calculate tonality statistics
        title_tonality = cluster_df['title_tonality'].value_counts()
        text_tonality = cluster_df['text_tonality'].value_counts()
        
        # Find most characteristic titles
        sample_titles = cluster_df['title'].sample(min(5, len(cluster_df))).tolist()
        
        # Store statistics
        cluster_stats[cluster_id] = {
            'size': len(cluster_df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'common_subjects': subjects.to_dict(),
            'common_sources': sources.to_dict(),
            'title_tonality': title_tonality.to_dict(),
            'text_tonality': text_tonality.to_dict(),
            'sample_titles': sample_titles
        }
        
        # Add feature importance if feature names are provided
        if feature_names is not None and hasattr(df, 'cluster_centers_'):
            # Find most important linguistic features
            center = df.cluster_centers_[cluster_id][:len(feature_names)]
            top_features_idx = center.argsort()[-top_n:][::-1]
            top_features = [(feature_names[i], center[i]) for i in top_features_idx]
            cluster_stats[cluster_id]['top_features'] = top_features
    
    return cluster_stats

def generate_cluster_report(cluster_stats, output_file='cluster_report.txt'):
    """Generate a human-readable report of cluster analysis"""
    with open(output_file, 'w') as f:
        f.write("=== CRYPTOCURRENCY NEWS CLUSTER ANALYSIS ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total clusters: {len(cluster_stats)}\n\n")
        
        for cluster_id, stats in sorted(cluster_stats.items()):
            # Use descriptive name if available
            cluster_name = stats.get('descriptive_name', f"Cluster {cluster_id}")
            
            f.write(f"--- CLUSTER {cluster_id}: {cluster_name} ---\n")
            f.write(f"Size: {stats['size']} articles\n\n")
            
            f.write("SENTIMENT:\n")
            f.write(f"  Average polarity: {stats['avg_polarity']:.2f}\n")
            f.write(f"  Average subjectivity: {stats['avg_subjectivity']:.2f}\n")
            f.write("  Sentiment distribution:\n")
            for sentiment, count in stats['sentiment_distribution'].items():
                f.write(f"    - {sentiment}: {count} articles ({count/stats['size']*100:.1f}%)\n")
            
            f.write("\nCOMMON SUBJECTS:\n")
            for subject, count in stats['common_subjects'].items():
                f.write(f"  - {subject}: {count} articles\n")
                
            f.write("\nCOMMON SOURCES:\n")
            for source, count in stats['common_sources'].items():
                f.write(f"  - {source}: {count} articles\n")
            
            f.write("\nTONALITY:\n")
            f.write("  Title tonality:\n")
            for tone, count in stats['title_tonality'].items():
                f.write(f"    - {tone}: {count} articles\n")
            f.write("  Text tonality:\n")
            for tone, count in stats['text_tonality'].items():
                f.write(f"    - {tone}: {count} articles\n")
            
            if 'top_features' in stats:
                f.write("\nDISTINGUISHING FEATURES:\n")
                for feature, value in stats['top_features']:
                    f.write(f"  - {feature}: {value:.3f}\n")
            
            f.write("\nSAMPLE TITLES:\n")
            for i, title in enumerate(stats['sample_titles']):
                f.write(f"  {i+1}. {title}\n")
            
            f.write("\n" + "-" * 40 + "\n\n")
    
    print(f"Cluster report generated: {output_file}")

def time_based_analysis(df, clusters):
    """Analyze how clusters evolve over time"""
    # Group by date and cluster, count articles
    time_cluster_counts = df.groupby([pd.Grouper(key='date', freq='W'), 'cluster']).size().unstack().fillna(0)
    
    # Plot time series of cluster sizes
    plt.figure(figsize=(15, 8))
    time_cluster_counts.plot(ax=plt.gca())
    plt.title('Cluster Evolution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('crypto_news_cluster_evolution.png')
    plt.close()
    
    return time_cluster_counts

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('/Users/stefanalexandru/Desktop/ARMS/crypto_nlp_results_intermediate.csv')
    
    # Create feature vectors
    features, feature_names = create_feature_vectors(df)
    
    # Determine optimal number of clusters
    inertia = []
    k_range = range(2, 15)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Curve for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('crypto_news_elbow_curve.png')
    
    # Choose optimal k from elbow curve (let's say k=5 for this example)
    optimal_k = 5  # This would be determined from the elbow curve
    
    # Perform clustering with optimal k
    kmeans_clusters, dbscan_clusters, kmeans_model = perform_clustering(features, n_clusters=optimal_k)
    
    # Visualize the clusters with enhanced plots
    pca_features = visualize_clusters(features, kmeans_clusters, "K-means Clusters", df)
    visualize_clusters(features, dbscan_clusters, "DBSCAN Clusters", df)
    
    # Analyze the clusters
    kmeans_stats = analyze_clusters(df, kmeans_clusters, feature_names)
    
    # Generate descriptive names for the clusters
    cluster_names = generate_cluster_names(kmeans_stats)
    
    # Add descriptive names to the cluster report
    for cluster_id, name in cluster_names.items():
        if cluster_id in kmeans_stats:
            kmeans_stats[cluster_id]['descriptive_name'] = name
    
    # Generate cluster report
    generate_cluster_report(kmeans_stats, 'crypto_news_kmeans_clusters.txt')
    
    # Perform time-based analysis
    time_analysis = time_based_analysis(df, kmeans_clusters)
    
    # Visualization of cluster characteristics
    
    # 1. Create a heatmap of sentiment by cluster
    sentiment_by_cluster = pd.crosstab(df['cluster'], df['sentiment_class'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(sentiment_by_cluster, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Sentiment Distribution by Cluster')
    plt.tight_layout()
    plt.savefig('crypto_news_sentiment_by_cluster.png')
    
    # 2. Create a heatmap of subject by cluster
    subject_by_cluster = pd.crosstab(df['cluster'], df['subject'])
    plt.figure(figsize=(14, 10))
    sns.heatmap(subject_by_cluster, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Subject Distribution by Cluster')
    plt.tight_layout()
    plt.savefig('crypto_news_subject_by_cluster.png')
    
    # 3. Plot average sentiment polarity and subjectivity by cluster
    sentiment_stats = df.groupby('cluster')[['sentiment_polarity', 'sentiment_subjectivity']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = sentiment_stats['cluster']
    width = 0.35
    ax.bar(x - width/2, sentiment_stats['sentiment_polarity'], width, label='Polarity')
    ax.bar(x + width/2, sentiment_stats['sentiment_subjectivity'], width, label='Subjectivity')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Score')
    ax.set_title('Average Sentiment Metrics by Cluster')
    ax.legend()
    plt.tight_layout()
    plt.savefig('crypto_news_sentiment_metrics_by_cluster.png')
    
    # 4. Create scatter plot of articles colored by cluster with sentiment metrics
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df['sentiment_polarity'], 
        df['sentiment_subjectivity'], 
        c=df['cluster'], 
        cmap='viridis', 
        alpha=0.6
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Sentiment Subjectivity')
    plt.title('Article Distribution by Sentiment and Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('crypto_news_sentiment_scatter.png')
    
    # 5. Create a visualization of linguistic features by cluster
    linguistic_counts = df.groupby('cluster')[[
        'title_verbs_count', 'title_adjectives_count',
        'text_verbs_count', 'text_adjectives_count'
    ]].mean()
    
    linguistic_counts.plot(kind='bar', figsize=(14, 8))
    plt.title('Average Linguistic Feature Counts by Cluster')
    plt.ylabel('Average Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('crypto_news_linguistic_counts.png')
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()