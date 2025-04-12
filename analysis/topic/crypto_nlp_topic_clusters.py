import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import json
import ast
from datetime import datetime
import re
import plotly.express as px
import plotly.graph_objects as go
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_and_clean_data(file_path):
    """Load and preprocess the cryptocurrency news data"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Parse sentiment column (as dictionary in string format)
    df['sentiment_parsed'] = df['sentiment'].apply(
        lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else {})
    
    # Extract sentiment features
    df['sentiment_class'] = df['sentiment_parsed'].apply(
        lambda x: x.get('class', 'unknown') if isinstance(x, dict) else 'unknown')
    df['sentiment_polarity'] = df['sentiment_parsed'].apply(
        lambda x: float(x.get('polarity', 0)) if isinstance(x, dict) else 0)
    df['sentiment_subjectivity'] = df['sentiment_parsed'].apply(
        lambda x: float(x.get('subjectivity', 0)) if isinstance(x, dict) else 0)
    
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
    
    # Parse linguistics columns and handle missing values
    for col in ['title_verbs', 'title_adjectives', 'text_verbs', 'text_adjectives']:
        df[f'{col}_parsed'] = df[col].apply(lambda x: safe_parse_json(x, []))
        df[f'{col}_count'] = df[f'{col}_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df[f'{col}_text'] = df[f'{col}_parsed'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    # Create a feature for tonality presence
    tonality_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['title_tonality_score'] = df['title_tonality'].map(tonality_map).fillna(0)
    df['text_tonality_score'] = df['text_tonality'].map(tonality_map).fillna(0)
    
    # Create a combined feature of all linguistic elements for topic modeling
    df['all_verbs'] = df.apply(lambda row: ' '.join(filter(None, [row['title_verbs_text'], row['text_verbs_text']])), axis=1)
    df['all_adjectives'] = df.apply(lambda row: ' '.join(filter(None, [row['title_adjectives_text'], row['text_adjectives_text']])), axis=1)
    
    return df

def extract_topic_features(df, n_topics=10, n_top_words=15):
    """Extract topics from the text data using NMF and LDA"""
    print(f"Extracting {n_topics} topics from linguistic features...")
    
    # For verbs
    verb_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    verb_dtm = verb_vectorizer.fit_transform(df['all_verbs'].fillna(''))
    
    # For adjectives
    adj_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    adj_dtm = adj_vectorizer.fit_transform(df['all_adjectives'].fillna(''))
    
    # NMF for verbs
    nmf_verb = NMF(n_components=n_topics, random_state=42)
    nmf_verb_features = nmf_verb.fit_transform(verb_dtm)
    
    # Column names for the features
    verb_topic_cols = [f'verb_topic_{i}' for i in range(n_topics)]
    df_verb_topics = pd.DataFrame(nmf_verb_features, columns=verb_topic_cols)
    
    # NMF for adjectives
    nmf_adj = NMF(n_components=n_topics, random_state=42)
    nmf_adj_features = nmf_adj.fit_transform(adj_dtm)
    
    adj_topic_cols = [f'adj_topic_{i}' for i in range(n_topics)]
    df_adj_topics = pd.DataFrame(nmf_adj_features, columns=adj_topic_cols)
    
    # Extract top words for each topic
    verb_feature_names = verb_vectorizer.get_feature_names_out()
    adj_feature_names = adj_vectorizer.get_feature_names_out()
    
    # Get top words for each verb topic
    verb_topic_words = {}
    for topic_idx, topic in enumerate(nmf_verb.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [verb_feature_names[i] for i in top_features_ind]
        verb_topic_words[f'verb_topic_{topic_idx}'] = top_features
    
    # Get top words for each adjective topic
    adj_topic_words = {}
    for topic_idx, topic in enumerate(nmf_adj.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [adj_feature_names[i] for i in top_features_ind]
        adj_topic_words[f'adj_topic_{topic_idx}'] = top_features
    
    # Combine the features
    df_combined = pd.concat([df.reset_index(drop=True), df_verb_topics, df_adj_topics], axis=1)
    
    return df_combined, verb_topic_words, adj_topic_words

def cluster_articles(df, n_clusters=5):
    """Cluster articles based on topic features and sentiment"""
    print(f"Clustering articles into {n_clusters} groups...")
    
    # Select features for clustering
    verb_topic_cols = [col for col in df.columns if col.startswith('verb_topic_')]
    adj_topic_cols = [col for col in df.columns if col.startswith('adj_topic_')]
    
    features = df[verb_topic_cols + adj_topic_cols + [
        'sentiment_polarity', 'sentiment_subjectivity', 
        'title_tonality_score', 'text_tonality_score'
    ]]
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['topic_cluster'] = kmeans.fit_predict(scaled_features)
    
    return df, kmeans.cluster_centers_

def generate_descriptive_cluster_names(df, cluster_insights):
    """Generate descriptive names for clusters based on their characteristics"""
    descriptive_names = {}
    
    for cluster_id, insights in cluster_insights.items():
        # Extract key characteristics
        dominant_sentiment = max(insights['sentiment'].items(), key=lambda x: x[1])[0]
        
        # Get the dominant subject
        dominant_subject = list(insights['subjects'].keys())[0] if insights['subjects'] else 'general'
        
        # Get dominant source
        dominant_source = list(insights['sources'].keys())[0] if insights['sources'] else 'various'
        
        # Get sentiment polarity and subjectivity
        polarity = insights['avg_polarity']
        subjectivity = insights['avg_subjectivity']
        
        # Extract key topics
        verb_topic_terms = []
        for topic, words in insights['top_verb_topics']:
            verb_topic_terms.extend(words[:3])  # Take top 3 words from each verb topic
            
        adj_topic_terms = []
        for topic, words in insights['top_adj_topics']:
            adj_topic_terms.extend(words[:3])  # Take top 3 words from each adjective topic
            
        # Create sentiment descriptor
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
            
        # Add subjectivity qualifier
        if subjectivity > 0.5:
            sentiment_desc = "Subjective " + sentiment_desc
        elif subjectivity < 0.2:
            sentiment_desc = "Objective " + sentiment_desc
            
        # Find most distinctive topics
        key_verbs = "/".join(verb_topic_terms[:2]) if verb_topic_terms else ""
        key_adjs = "/".join(adj_topic_terms[:2]) if adj_topic_terms else ""
            
        # Base name on subject and sentiment
        base_name = f"{sentiment_desc} {dominant_subject.title()} ({dominant_source})"
        
        # Add linguistic characteristics if available
        if key_verbs and key_adjs:
            descriptive_name = f"{base_name} - {key_adjs} {key_verbs}"
        else:
            descriptive_name = base_name
            
        # Truncate if too long
        if len(descriptive_name) > 60:
            descriptive_name = descriptive_name[:57] + "..."
            
        descriptive_names[cluster_id] = descriptive_name
        
    return descriptive_names

def analyze_topic_clusters(df, verb_topic_words, adj_topic_words):
    """Analyze the characteristics of each topic cluster"""
    print("Analyzing topic clusters...")
    clusters = df['topic_cluster'].unique()
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs('topic_cluster_visualizations', exist_ok=True)
    
    cluster_insights = {}
    
    for cluster in sorted(clusters):
        cluster_df = df[df['topic_cluster'] == cluster]
        
        # Size of the cluster
        cluster_size = len(cluster_df)
        
        # Most common subjects
        subject_counts = cluster_df['subject'].value_counts().head(5).to_dict()
        
        # Sentiment distribution
        sentiment_counts = cluster_df['sentiment_class'].value_counts().to_dict()
        
        # Average sentiment metrics
        avg_polarity = cluster_df['sentiment_polarity'].mean()
        avg_subjectivity = cluster_df['sentiment_subjectivity'].mean()
        
        # Most common sources
        source_counts = cluster_df['source'].value_counts().head(3).to_dict()
        
        # Time distribution
        time_dist = cluster_df.groupby(pd.Grouper(key='date', freq='W')).size()
        
        # Most dominant verb topics
        verb_topic_cols = [col for col in df.columns if col.startswith('verb_topic_')]
        avg_verb_topics = cluster_df[verb_topic_cols].mean().sort_values(ascending=False)
        top_verb_topics = avg_verb_topics.head(3).index.tolist()
        
        # Most dominant adjective topics
        adj_topic_cols = [col for col in df.columns if col.startswith('adj_topic_')]
        avg_adj_topics = cluster_df[adj_topic_cols].mean().sort_values(ascending=False)
        top_adj_topics = avg_adj_topics.head(3).index.tolist()
        
        # Sample titles
        sample_titles = cluster_df['title'].sample(min(5, len(cluster_df))).tolist()
        
        # Store insights
        cluster_insights[cluster] = {
            'size': cluster_size,
            'subjects': subject_counts,
            'sentiment': sentiment_counts,
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'sources': source_counts,
            'top_verb_topics': [(topic, verb_topic_words[topic]) for topic in top_verb_topics],
            'top_adj_topics': [(topic, adj_topic_words[topic]) for topic in top_adj_topics],
            'sample_titles': sample_titles
        }
        
        # Create visualizations for this cluster
        
        # 1. Subject distribution
        subject_df = pd.DataFrame(list(subject_counts.items()), columns=['Subject', 'Count'])
        if not subject_df.empty:
            fig = px.bar(subject_df, x='Subject', y='Count', 
                        title=f'Subject Distribution - Cluster {cluster}')
            fig.write_html(f'topic_cluster_visualizations/cluster_{cluster}_subjects.html')
        
        # 2. Sentiment distribution
        sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])
        if not sentiment_df.empty:
            fig = px.pie(sentiment_df, values='Count', names='Sentiment', 
                        title=f'Sentiment Distribution - Cluster {cluster}')
            fig.write_html(f'topic_cluster_visualizations/cluster_{cluster}_sentiment.html')
        
        # 3. Time series
        if len(time_dist) > 1:
            fig = px.line(x=time_dist.index, y=time_dist.values, 
                        title=f'Articles Over Time - Cluster {cluster}')
            fig.update_layout(xaxis_title='Date', yaxis_title='Article Count')
            fig.write_html(f'topic_cluster_visualizations/cluster_{cluster}_time.html')
        
    return cluster_insights

def generate_topic_cluster_report(cluster_insights, output_file='topic_cluster_report.html'):
    """Generate an HTML report of the topic cluster analysis"""
    print("Generating topic cluster report...")
    
    # Generate descriptive names for clusters
    descriptive_names = generate_descriptive_cluster_names(None, cluster_insights)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto News Topic Cluster Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
            h3 { color: #2980b9; }
            .cluster { background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
            .stats { margin-left: 20px; }
            .topic-words { background-color: #e9ecef; padding: 10px; border-radius: 3px; }
            .sample { font-style: italic; color: #555; }
        </style>
    </head>
    <body>
        <h1>Cryptocurrency News Topic Cluster Analysis</h1>
        <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        <p>Total clusters: """ + str(len(cluster_insights)) + """</p>
    """
    
    for cluster, insights in sorted(cluster_insights.items()):
        cluster_name = descriptive_names.get(cluster, f"Cluster {cluster}")
        
        html_content += f"""
        <div class="cluster">
            <h2>Cluster {cluster}: {cluster_name} ({insights['size']} articles)</h2>
            
            <h3>Top Verb Topics</h3>
            <div class="stats">
        """
        
        for topic, words in insights['top_verb_topics']:
            html_content += f"""
                <p><strong>{topic}:</strong></p>
                <p class="topic-words">{', '.join(words)}</p>
            """
        
        html_content += """
            </div>
            
            <h3>Top Adjective Topics</h3>
            <div class="stats">
        """
        
        for topic, words in insights['top_adj_topics']:
            html_content += f"""
                <p><strong>{topic}:</strong></p>
                <p class="topic-words">{', '.join(words)}</p>
            """
        
        html_content += f"""
            </div>
            
            <h3>Common Subjects</h3>
            <div class="stats">
                <ul>
        """
        
        for subject, count in insights['subjects'].items():
            html_content += f"<li>{subject}: {count} articles</li>\n"
        
        html_content += """
                </ul>
            </div>
            
            <h3>Sentiment</h3>
            <div class="stats">
                <p>Average polarity: {:.2f}</p>
                <p>Average subjectivity: {:.2f}</p>
                <ul>
        """.format(insights['avg_polarity'], insights['avg_subjectivity'])
        
        for sentiment, count in insights['sentiment'].items():
            pct = (count / insights['size']) * 100
            html_content += f"<li>{sentiment}: {count} articles ({pct:.1f}%)</li>\n"
        
        html_content += """
                </ul>
            </div>
            
            <h3>Sample Article Titles</h3>
            <div class="stats">
                <ol>
        """
        
        for title in insights['sample_titles']:
            html_content += f"<li class='sample'>{title}</li>\n"
        
        html_content += """
                </ol>
            </div>
            
            <h3>Visualizations</h3>
            <div class="stats">
                <ul>
                    <li><a href="topic_cluster_visualizations/cluster_{}_subjects.html" target="_blank">Subject Distribution</a></li>
                    <li><a href="topic_cluster_visualizations/cluster_{}_sentiment.html" target="_blank">Sentiment Distribution</a></li>
                    <li><a href="topic_cluster_visualizations/cluster_{}_time.html" target="_blank">Articles Over Time</a></li>
                </ul>
            </div>
        </div>
        """.format(cluster, cluster, cluster)
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Topic cluster report generated: {output_file}")

def visualize_overall_results(df, verb_topic_words, adj_topic_words):
    """Create overall visualizations of the topic modeling results"""
    print("Generating overall visualizations...")
    
    os.makedirs('topic_cluster_visualizations', exist_ok=True)
    
    # Generate descriptive cluster names
    cluster_insights = analyze_topic_clusters(df, verb_topic_words, adj_topic_words)
    descriptive_names = generate_descriptive_cluster_names(df, cluster_insights)
    
    # Map cluster IDs to descriptive names
    cluster_name_map = {k: f"Cluster {k}: {v}" for k, v in descriptive_names.items()}
    
    # Add descriptive names to the DataFrame
    df['cluster_name'] = df['topic_cluster'].map(cluster_name_map)
    
    # 1. Distribution of articles across clusters with descriptive names
    cluster_counts = df['topic_cluster'].value_counts().sort_index()
    cluster_labels = [cluster_name_map.get(i, f"Cluster {i}") for i in cluster_counts.index]
    
    fig = px.bar(x=cluster_labels, y=cluster_counts.values,
               labels={'x': 'Cluster', 'y': 'Number of Articles'},
               title='Distribution of Articles Across Topic Clusters')
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html('topic_cluster_visualizations/cluster_distribution.html')
    
    # 2. Subject distribution by cluster
    subject_by_cluster = pd.crosstab(df['topic_cluster'], df['subject'])
    # Convert to percentage
    subject_by_cluster_pct = subject_by_cluster.div(subject_by_cluster.sum(axis=1), axis=0) * 100
    
    # Select top subjects (columns with highest totals)
    top_subjects = subject_by_cluster.sum().sort_values(ascending=False).head(10).index
    selected_data = subject_by_cluster_pct[top_subjects]
    
    # Create heatmap
    fig = px.imshow(selected_data.T, 
                   labels=dict(x="Cluster", y="Subject", color="Percentage"),
                   title="Subject Distribution by Topic Cluster (%)")
    fig.write_html('topic_cluster_visualizations/subject_by_cluster_heatmap.html')
    
    # 3. Sentiment by cluster
    sentiment_by_cluster = pd.crosstab(df['topic_cluster'], df['sentiment_class'])
    sentiment_by_cluster_pct = sentiment_by_cluster.div(sentiment_by_cluster.sum(axis=1), axis=0) * 100
    
    fig = px.imshow(sentiment_by_cluster_pct.T, 
                   labels=dict(x="Cluster", y="Sentiment", color="Percentage"),
                   title="Sentiment Distribution by Topic Cluster (%)")
    fig.write_html('topic_cluster_visualizations/sentiment_by_cluster_heatmap.html')
    
    # 4. Create an interactive 3D scatter plot based on top 3 PCA components
    from sklearn.decomposition import PCA
    
    # Prepare features
    verb_topic_cols = [col for col in df.columns if col.startswith('verb_topic_')]
    adj_topic_cols = [col for col in df.columns if col.startswith('adj_topic_')]
    
    features = df[verb_topic_cols + adj_topic_cols]
    
    # Apply PCA for 3D visualization
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'PCA3': pca_result[:, 2],
        'Cluster': df['topic_cluster'],
        'Subject': df['subject'],
        'Title': df['title'],
        'Sentiment': df['sentiment_class']
    })
    
    # Create the 3D scatter plot
    fig = px.scatter_3d(
        pca_df, x='PCA1', y='PCA2', z='PCA3',
        color='Cluster', symbol='Subject', hover_data=['Title', 'Sentiment'],
        title='3D PCA Visualization of Topic Clusters',
        labels={'Cluster': 'Topic Cluster'}
    )
    
    fig.update_layout(scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ))
    
    fig.write_html('topic_cluster_visualizations/3d_pca_visualization.html')
    
    # 5. Time series of clusters with descriptive names
    time_series = df.groupby([pd.Grouper(key='date', freq='W'), 'topic_cluster']).size().reset_index(name='count')
    
    # Merge with descriptive names
    time_series['cluster_name'] = time_series['topic_cluster'].map(cluster_name_map)
    
    fig = px.line(time_series, x='date', y='count', color='cluster_name', 
                 labels={'date': 'Date', 'count': 'Number of Articles', 'cluster_name': 'Topic Cluster'},
                 title='Topic Clusters Over Time')
    fig.write_html('topic_cluster_visualizations/clusters_time_series.html')

def run_analysis(file_path, n_topics=12, n_clusters=6):
    """Run the complete topic modeling and clustering analysis"""
    # Load and clean the data
    df = load_and_clean_data(file_path)
    
    # Extract topic features
    df_with_topics, verb_topic_words, adj_topic_words = extract_topic_features(df, n_topics)
    
    # Cluster the articles based on topics
    clustered_df, cluster_centers = cluster_articles(df_with_topics, n_clusters)
    
    # Analyze the topic clusters
    cluster_insights = analyze_topic_clusters(clustered_df, verb_topic_words, adj_topic_words)
    
    # Generate descriptive names
    descriptive_names = generate_descriptive_cluster_names(clustered_df, cluster_insights)
    
    # Add descriptive names to DataFrame
    cluster_name_map = {k: v for k, v in descriptive_names.items()}
    clustered_df['cluster_name'] = clustered_df['topic_cluster'].map(cluster_name_map)
    
    # Generate an HTML report
    generate_topic_cluster_report(cluster_insights, 'topic_cluster_report.html')
    
    # Create overall visualizations
    visualize_overall_results(clustered_df, verb_topic_words, adj_topic_words)
    
    print("Topic modeling and clustering analysis completed successfully!")
    print("View the HTML report and visualizations in the topic_cluster_visualizations directory.")
    
    return clustered_df, cluster_insights, descriptive_names

if __name__ == "__main__":
    # Run the analysis
    result_df, insights, descriptive_names = run_analysis('/Users/stefanalexandru/Desktop/ARMS/crypto_nlp_results_intermediate.csv')
    
    # Save the results to CSV for further analysis
    result_df.to_csv('crypto_news_with_topic_clusters.csv', index=False)
    print("Results saved to crypto_news_with_topic_clusters.csv")