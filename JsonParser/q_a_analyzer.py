import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime, date
from math import sqrt
from scipy import stats

def diff_days(year, month, day):
    d = date(year, month, day)
    return (datetime.utcnow().date() - d).days

def extract_field_from_jsons(folder_path, field_name):
    values = []
    folder = Path(folder_path)
    for i in range(150):
        json_file = folder / f"{i}.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding='utf-8'))
                posts = data.get('data', {}).get('post_stream', {}).get('posts', [])
                for post in posts:
                    if field_name in post and post[field_name] is not None:
                        values.append(post[field_name])
                print(f"Processed {json_file.name} - Found {len([p for p in posts if field_name in p])} posts with {field_name}")
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
        else:
            print(f"File {json_file.name} not found")
    return values

def extract_reactions_from_jsons(folder_path):
    folder = Path(folder_path)
    positive_ids = {"heart", "point_up", "+1", "laughing", "call_me_hand", "hugs"}
    negative_ids = {"-1", "question", "cry", "angry"}
    counts = Counter()
    pos_dist, neg_dist = [], []
    for i in range(150):
        json_file = folder / f"{i}.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding='utf-8'))
                posts = data.get('data', {}).get('post_stream', {}).get('posts', [])
                for post in posts:
                    ps, ns = 0, 0
                    for r in post.get('reactions', []):
                        counts[r['id']] += r['count']
                        if r['id'] in positive_ids:
                            ps += r['count']
                        elif r['id'] in negative_ids:
                            ns += r['count']
                    pos_dist.append(ps)
                    neg_dist.append(ns)
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
    return counts, pos_dist, neg_dist

def extract_metric_from_jsons(folder_path):
    folder = Path(folder_path)
    metrics = []
    SCORE_CLIPPING = 1000
    RECENCY_DECAY = 1080
    MIN_TRUST_LEVEL = 0.1
    Q_A_CLIPPING = 1500
    for i in range(150):
        json_file = folder / f"{i}.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding='utf-8'))
                posts = data.get('data', {}).get('post_stream', {}).get('posts', [])
                for post in posts:
                    if all(k in post for k in ('score','created_at','readers_count','trust_level')):
                        clipped = sqrt(min(post['score'], SCORE_CLIPPING))
                        y,m,d = map(int, post['created_at'].split('T')[0].split('-'))
                        diff = diff_days(y,m,d)
                        recency = np.exp(- diff / RECENCY_DECAY)
                        confidence = sqrt(post['readers_count'])
                        metric = sqrt(min(recency * confidence * clipped * (post['trust_level'] + MIN_TRUST_LEVEL), Q_A_CLIPPING))
                        metrics.append(metric)
                        if metric > 50000:
                            print("\n=== High Q_A Metric Detected ===")
                            print(f"File: {json_file.name}")
                            print(f"Created at: {post['created_at']}")
                            print(f"Score (clipped): {clipped}")
                            print(f"Readers count: {post['readers_count']}")
                            print(f"Trust level: {post['trust_level']}")
                            print(f"Recency factor: {recency:.4f}")
                            print(f"Confidence: {confidence:.2f}")
                            print(f"Computed Q_A metric: {metric:.2f}")
                            print(f"Post content (cooked):\n{post.get('cooked','[No content]')}")
                            reason = []
                            if clipped > 900:
                                reason.append("High clipped score")
                            if confidence > 50:
                                reason.append("Large readership")
                            if recency > 0.9:
                                reason.append("Very recent post")
                            if post['trust_level'] > 2:
                                reason.append("High trust level")
                            print(f"Potential Reasons: {', '.join(reason) if reason else 'Unusual combination of factors'}\n")
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
    return metrics

def plot_metric_distribution(metrics, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    axs[0].hist(metrics, bins=50, alpha=0.7, edgecolor='black')
    axs[0].set_title('Metric Histogram')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True, alpha=0.3)
    axs[0].axvline(np.mean(metrics), color='red', linestyle='--', label=f'Mean: {np.mean(metrics):.2f}')
    axs[0].legend()
    axs[1].scatter(range(len(metrics)), metrics, alpha=0.6, s=10)
    axs[1].set_title('Metric Scatter')
    axs[1].set_xlabel('Post Index')
    axs[1].set_ylabel('Metric')
    axs[1].grid(True, alpha=0.3)
    density = stats.gaussian_kde(metrics)
    xs = np.linspace(min(metrics), max(metrics), 200)
    axs[2].plot(xs, density(xs), linewidth=2)
    axs[2].set_title('Metric Density')
    axs[2].set_xlabel('Value')
    axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cumulative_score(scores, save_path=None):
    clipped_scores = [min(score, 1000) for score in scores]
    sorted_scores = np.sort(clipped_scores)[::-1]
    cumulative = np.cumsum(sorted_scores)
    cumulative_norm = cumulative / cumulative[-1]
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_norm, linewidth=2, color='blue')
    plt.title('Cumulative Sum of Scores (Clipped at 1000) Normalized to 1', fontsize=14, fontweight='bold')
    plt.xlabel('Posts sorted by score (descending)')
    plt.ylabel('Cumulative fraction of total score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cumulative_metric(metrics, save_path=None):
    clipped_metrics = [min(m, 5000) for m in metrics]
    sorted_metrics = np.sort(clipped_metrics)
    cumulative = np.cumsum(sorted_metrics)
    cumulative_norm = cumulative / cumulative[-1]
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_norm, linewidth=2, color='green')
    plt.title('Cumulative Sum of Q_A Metric (Clipped at 5000) Normalized to 1', fontsize=14, fontweight='bold')
    plt.xlabel('Posts sorted by metric (ascending)')
    plt.ylabel('Cumulative fraction of total metric')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    folder = "Cheif_Delphi_Jsons"
    scores = extract_field_from_jsons(folder, 'score')
    readers = extract_field_from_jsons(folder, 'readers_count')
    trust = extract_field_from_jsons(folder, 'trust_level')
    reactions, pos, neg = extract_reactions_from_jsons(folder)
    metric = extract_metric_from_jsons(folder)
    print(f"Scores: {len(scores)}, Readers: {len(readers)}, Trust: {len(trust)}, Metric: {len(metric)}")
    plot_metric_distribution(metric, save_path="metric_dist.png")
    plot_cumulative_score(scores, save_path="chief_delphi_cumulative_score.png")
    plot_cumulative_metric(metric, save_path="chief_delphi_cumulative_metric.png")

if __name__ == "__main__":
    main()