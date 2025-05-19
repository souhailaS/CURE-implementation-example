import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import hdbscan

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_csv_numerical(file_path):
    try:
        data = []
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)

            for row in reader:
                data.append(row)

        df = pd.DataFrame(data, columns=header)
        print(f"\nLoaded {len(df)} rows from the CSV file.")
        print("DataFrame Preview:")
        print(df.head())
        return df

    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def calculate_difference_vector(df):
    try:
        df['commit_date'] = pd.to_datetime(df['commit_date'], errors='coerce')
        df = df.dropna(subset=['commit_date'])
        df = df.sort_values(['api_spec_id', 'commit_date'])
        difference_vectors = []

        for api_spec_id, group in df.groupby('api_spec_id'):
            first_commit = group.iloc[0]
            last_commit = group.iloc[-1]
            columns_to_exclude = ['api_spec_id', 'api_version', 'commits', 'commit_date'] # should we keep the commits for the clustering ??
            numeric_first = pd.to_numeric(first_commit.drop(columns_to_exclude, errors='ignore'), errors='coerce').fillna(0)
            numeric_last = pd.to_numeric(last_commit.drop(columns_to_exclude, errors='ignore'), errors='coerce').fillna(0)
            
            # this is the difference vector that represent each API.
            difference_vector = numeric_last - numeric_first
            difference_vector['api_spec_id'] = api_spec_id
            difference_vector['commit_date'] = last_commit['commit_date']
            difference_vectors.append(difference_vector)

        diff_df = pd.DataFrame(difference_vectors)
        print(f"\nCalculated difference vectors for {len(diff_df)} APIs.")
        return diff_df

    except Exception as e:
        print(f"Error calculating difference vectors: {e}")
        return None

def cluster_and_save(df):
    columns_to_exclude = ['api_spec_id', 'api_version', 'commits', 'commit_date']
    clustering_data = df.drop(columns=columns_to_exclude, errors='ignore')
    clustering_data = clustering_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = clustering_data.values

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.5)
    labels = clusterer.fit_predict(X)
    df['cluster_number'] = labels

    print("\nHDBSCAN Clustering Results:")
    print(f"Number of clusters (excluding noise): {len(set(labels)) - (1 if -1 in labels else 0)}")
    print("Cluster Labels:")
    print(labels)
    print("\nCluster Information:")
    for cluster_label in set(labels):
        count = sum(labels == cluster_label)
        if cluster_label == -1:
            print(f"Noise: {count} points")
        else:
            print(f"Cluster {cluster_label}: {count} points")

    output_df = df[['api_spec_id', 'cluster_number', 'api_version', 'commits', 'commit_date']]
    output_df.to_csv('api_clustering_all_commits.csv', index=False)
    print("\nClustered data saved to 'api_clustering_all_commits.csv'")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_points = X_tsne[labels == label]
        if label == -1:
            color = 'gray'
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, s=1, label="Noise", alpha=0.7)
        else:
            color = plt.cm.get_cmap('viridis')(float(label) / (len(unique_labels) - 1))
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=1, label=f'Cluster {label}', alpha=0.7)

    plt.title('API Clustering with HDBSCAN and t-SNE (All Commits)', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', markerscale=10)
    plt.show()

file_path = 'oas.commits.metrics.csv'
df = read_csv_numerical(file_path)

if df is not None:
    cluster_and_save(df)