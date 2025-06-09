import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from Levenshtein import distance as Levenshtein_distance
from datetime import datetime
import shutil
from collections import Counter

def create_output_dir():
    """
    Создает директорию для результатов с временной меткой
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = os.path.join('output', f'run_{timestamp}')
    os.makedirs(output_dir)
    
    return output_dir

def plot_pattern_weight(patterns: List[Dict], top_n: int = 10, output_dir: str = None):
    """
    Визуализация веса паттернов
    
    Args:
        patterns: Список паттернов
        top_n: Количество паттернов с наибольшим весом для отображения
        output_dir: Директория для сохранения результатов
    """
    if output_dir is None:
        output_dir = create_output_dir()

    df = pd.DataFrame(patterns)
    df = df.sort_values('weight', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 12))
    bars = plt.bar(range(len(df)), df['weight'])
    plt.xticks(range(len(df)), [str(p) for p in df['pattern']], rotation=45, ha='right')
    plt.title('Топ {} паттернов с наибольшим весом'.format(top_n))
    plt.ylabel('Вес')
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    output_file = os.path.join(output_dir, 'pattern_weight.png')
    plt.savefig(output_file)
    plt.close()
    return output_file

def visualize_pattern_network(patterns: list[dict],
                              min_weight: float = .1,
                              coef_weight: float = 1e-4,
                              output_dir: str | None = None):
    """
    Голубые узлы, оранжевые ореолы плотности входящих рёбер,
    классические стрелки-переходы.
    """
    if output_dir is None:
        output_dir = create_output_dir()

    G = nx.DiGraph()
    for p in patterns:
        if p["weight"] < min_weight:
            continue
        w = p["weight"] * coef_weight
        for a, b in zip(p["pattern"][:-1], p["pattern"][1:]):
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w)

    in_w = Counter({v: 0.0 for v in G.nodes})
    for _, tgt, d in G.edges(data=True):
        in_w[tgt] += d["weight"]

    n_nodes  = len(G)
    k_layout = 150 / n_nodes**0.5
    pos = nx.spring_layout(G, k=k_layout, iterations=70, seed=1, scale=3.2)

    fig, ax = plt.subplots(figsize=(46, 46))

    base = 30_000
    nx.draw_networkx_nodes(
        G, pos, node_color="#90caf9", edgecolors="white",
        node_size=base, linewidths=1.4, ax=ax
    )

    for v, (x, y) in pos.items():
        node_sizes = [base + 0.5 * in_w[v] for v in G.nodes]
        halo_r   = (node_sizes[list(G.nodes).index(v)] ** 0.5) * 0.001
        halo_w   = halo_r * 500
        circle   = plt.Circle((x, y), radius=halo_r,
                              edgecolor="#ff6d00", linewidth=halo_w,
                              fill=False, alpha=.35)
        ax.add_patch(circle)

    weights = [d["weight"] * coef_weight for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle="->", arrowsize=23,
        width=weights, edge_color="#555555", alpha=.9, ax=ax
    )

    nx.draw_networkx_labels(G, pos, ax=ax)

    ax.set_title("Сеть последовательностей действий (плотность входов)",
                 fontsize=22)
    ax.axis("off")
    fig.tight_layout()

    out = os.path.join(output_dir, "pattern_network.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

def cluster_patterns(patterns: List[Dict], n_clusters: int = 3, output_dir: str = None):
    """
    Кластеризация паттернов с помощью K-means
    
    Args:
        patterns: Список паттернов
        n_clusters: Количество кластеров
        output_dir: Директория для сохранения результатов
    """
    if output_dir is None:
        output_dir = create_output_dir()

    vectorizer = CountVectorizer()
    pattern_strings = [' '.join(map(str, p['pattern'])) for p in patterns]
    X = vectorizer.fit_transform(pattern_strings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    df = pd.DataFrame({
        'pattern': [str(p['pattern']) for p in patterns],
        'weight': [p['weight'] for p in patterns],
        'cluster': clusters
    })

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='polar')

    N = len(df)
    theta = 2 * np.pi * (df.index / N)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    radius = df['pattern'].apply(lambda s: s.count(',') + 1)

    size = 80 + 220 * (df['weight'] / df['weight'].max())

    for cl in range(n_clusters):
        mask = df['cluster'] == cl
        ax.scatter(theta[mask],
                   radius[mask],
                   s=size[mask],
                   label=f'Кластер {cl}',
                   alpha=0.80,
                   linewidths=0.5,
                   edgecolors='black')

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.10),
              frameon=False, title='Кластеры')
    
    output_file = os.path.join(output_dir, 'pattern_clusters_KMeans.png')
    plt.savefig(output_file)
    plt.close()

    cluster_results_file = os.path.join(output_dir, 'kmeans_clusters.csv')
    df.to_csv(cluster_results_file, index=False)
    
    return df, output_file

def analyze_pattern_statistics(patterns: List[Dict], output_dir: str = None):
    """
    Статистический анализ паттернов
    
    Args:
        patterns: Список паттернов
        output_dir: Директория для сохранения результатов
    """
    if output_dir is None:
        output_dir = create_output_dir()
    
    df = pd.DataFrame(patterns)

    stats = {
        'total_patterns': len(patterns),
        'avg_weight': df['weight'].mean(),
        'max_weight': df['weight'].max(),
        'min_weight': df['weight'].min(),
        'avg_pattern_length': df['pattern'].apply(len).mean(),
        'max_pattern_length': df['pattern'].apply(len).max(),
        'min_pattern_length': df['pattern'].apply(len).min()
    }

    stats_df = pd.DataFrame([stats])

    stats_file = os.path.join(output_dir, 'pattern_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    
    return stats_df, stats_file

def cluster_patterns_by_levenshtein(patterns: List[Dict], max_distance: float = 5.0, COL_clusters: int = 10, output_dir: str = None):
    """
    Кластеризация паттернов по расстоянию Левенштейна
    
    Args:
        patterns: Список паттернов
        max_distance: Максимальное расстояние в одном кластере
        COL_clusters: Количество кластеров
        output_dir: Директория для сохранения результатов
    """
    if output_dir is None:
        output_dir = create_output_dir()

    pattern_strings = [' '.join(p['pattern']) for p in patterns]
    n = len(pattern_strings)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = Levenshtein_distance(pattern_strings[i], pattern_strings[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=max_distance,
        n_clusters=COL_clusters
    )
    labels = clustering.fit_predict(dist_matrix)

    df = pd.DataFrame({
        'pattern': [str(p['pattern']) for p in patterns],
        'weight': [p['weight'] for p in patterns],
        'cluster': labels
    })
    

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='polar')

    N = len(df)
    theta  = 2 * np.pi * (df.index / N)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    radius = df['pattern'].apply(lambda s: s.count(',') + 1)
    size   = 80 + 220 * (df['weight'] / df['weight'].max())

    for cl in range(COL_clusters):
        mask = df['cluster'] == cl
        ax.scatter(theta[mask],
                   radius[mask],
                   s=size[mask],
                   label=f'Кластер {cl}',
                   alpha=0.80,
                   linewidths=0.5,
                   edgecolors='black')

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.10),
              frameon=False, title='Кластеры')
    
    output_file = os.path.join(output_dir, 'pattern_clusters_levenshtein.png')
    plt.savefig(output_file)
    plt.close()

    cluster_results_file = os.path.join(output_dir, 'levenshtein_clusters.csv')
    df.to_csv(cluster_results_file, index=False)

    dist_matrix_file = os.path.join(output_dir, 'levenshtein_distance_matrix.csv')
    pd.DataFrame(dist_matrix).to_csv(dist_matrix_file, index=False)
    
    return df, dist_matrix, output_file 