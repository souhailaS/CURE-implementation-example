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
            
            # reach only the first 1000 rows
            # for i, row in enumerate(reader):
            #     if i < 1000:
            #         data.append(row)
            #     else:
            #         break
        if not data:
            print("No data found in the CSV file.")
            return None

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
        
        # size
        print(f"\nDataFrame before dropping rows with invalid 'commit_date': {len(df)} rows.")
        df = df.dropna(subset=['commit_date'])
        print(f"DataFrame after dropping rows with invalid 'commit_date': {len(df)} rows remaining.")
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

def cluster_apis(df):
    columns_to_exclude = ['api_spec_id', 'commit_date']
    clustering_data = df.drop(columns=columns_to_exclude, errors='ignore')
    clustering_data = clustering_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = clustering_data.values

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.5)
    labels = clusterer.fit_predict(X)
    df['cluster_number'] = labels
    return df[['api_spec_id', 'cluster_number']]  

def cluster_and_save(df, cluster_df):
    cluster_mapping = cluster_df.set_index('api_spec_id')['cluster_number'].to_dict()
    df['cluster_number'] = df['api_spec_id'].map(cluster_mapping).fillna(-1).astype(int)

    print("\nHDBSCAN Clustering Results:")
    unique_clusters = df['cluster_number'].unique()
    print(f"Number of clusters (excluding noise): {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    for cluster_label in unique_clusters:
        count = sum(df['cluster_number'] == cluster_label)
        if cluster_label == -1:
            print(f"Noise: {count} points")
        else:
            print(f"Cluster {cluster_label}: {count} points")

    output_df = df[['api_spec_id', 'cluster_number', 'api_version', 'commits', 'commit_date']]
    output_df.to_csv('api_clustering_diff_vector.csv', index=False)
    print("\nClustered data saved to 'api_clustering_diff_vector.csv'")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X = df.drop(columns=['api_spec_id', 'api_version', 'commits', 'commit_date', 'cluster_number'], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    unique_labels = set(df['cluster_number'])
    for label in unique_labels:
        cluster_points = X_tsne[df['cluster_number'] == label]
        color = 'gray' if label == -1 else plt.cm.get_cmap('viridis')(float(label) / (len(unique_labels) - 1))
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=1, label=f'Cluster {label}', alpha=0.7)

    plt.title('API Clustering with HDBSCAN and t-SNE (Difference Vectors)', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', markerscale=10)
    plt.show()

file_path = '/Users/souhailaserbout/Documents/oas.commits.metrics.csv'
df = read_csv_numerical(file_path)

if df is not None:
    diff_df = calculate_difference_vector(df)
    if diff_df is not None:
        cluster_df = cluster_apis(diff_df)
        cluster_and_save(df, cluster_df)