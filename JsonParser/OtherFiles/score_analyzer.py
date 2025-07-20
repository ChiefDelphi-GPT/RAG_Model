import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

def extract_field_from_jsons(folder_path, field_name):
    values = []
    folder = Path(folder_path)

    for i in range(150):
        json_file = folder / f"{i}.json"

        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'data' in data and 'post_stream' in data['data'] and 'posts' in data['data']['post_stream']:
                    posts = data['data']['post_stream']['posts']

                    for post in posts:
                        if field_name in post and post[field_name] is not None:
                            values.append(post[field_name])

                print(f"Processed {json_file.name} - Found {len([p for p in data['data']['post_stream']['posts'] if field_name in p])} posts with {field_name}")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {json_file.name}: {e}")
            except KeyError as e:
                print(f"Missing expected key in {json_file.name}: {e}")
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
        else:
            print(f"File {json_file.name} not found")

    return values

def extract_reactions_from_jsons(folder_path):
    folder = Path(folder_path)
    positive_ids = {"heart", "point_up", "+1", "laughing", "call_me_hand", "hugs"}
    negative_ids = {"-1", "question", "cry", "angry"}

    reaction_counts = Counter()
    positive_distribution = []
    negative_distribution = []

    for i in range(150):
        json_file = folder / f"{i}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'data' in data and 'post_stream' in data['data'] and 'posts' in data['data']['post_stream']:
                    posts = data['data']['post_stream']['posts']

                    for post in posts:
                        pos_sum = 0
                        neg_sum = 0
                        if 'reactions' in post:
                            for r in post['reactions']:
                                reaction_counts[r['id']] += r['count']
                                if r['id'] in positive_ids:
                                    pos_sum += r['count']
                                elif r['id'] in negative_ids:
                                    neg_sum += r['count']
                        positive_distribution.append(pos_sum)
                        negative_distribution.append(neg_sum)
            except Exception as e:
                print(f"Failed on file {json_file.name}: {e}")

    return reaction_counts, positive_distribution, negative_distribution

def plot_all_together_dual(scores, reader_counts, trust_levels, save_path=None):
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np

    fig, axs = plt.subplots(4, 3, figsize=(24, 20))
    fig.suptitle('Chief Delphi Post Analysis: Score vs Readers Count vs Trust Level', fontsize=18, fontweight='bold')

    datasets = [(scores, 'Score'), (reader_counts, 'Readers Count'), (trust_levels, 'Trust Level')]

    for col, (data, label) in enumerate(datasets):
        data = np.array(data)
        stats_col = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
        }

        mean = stats_col['mean']
        median = stats_col['median']
        std = stats_col['std']

        axs[0, col].scatter(range(len(data)), data, alpha=0.6, s=20)
        axs[0, col].set_title(f'{label} - Scatter Plot')
        axs[0, col].set_xlabel('Post Index')
        axs[0, col].set_ylabel(label)
        axs[0, col].grid(True, alpha=0.3)
        text = (
            f"Mean: {mean:.2f}\n"
            f"Median: {median:.2f}\n"
            f"Std: {std:.2f}\n"
            f"Min: {stats_col['min']:.2f}\n"
            f"Max: {stats_col['max']:.2f}"
        )
        axs[0, col].text(0.02, 0.98, text, transform=axs[0, col].transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        axs[1, col].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axs[1, col].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        axs[1, col].axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
        axs[1, col].legend()
        axs[1, col].set_title(f'{label} - Histogram')
        axs[1, col].set_xlabel(label)
        axs[1, col].set_ylabel('Frequency')
        axs[1, col].grid(True, alpha=0.3)

        jitter = np.random.normal(0, 0.05, len(data))
        axs[2, col].scatter(data, np.zeros_like(data), alpha=0.6, s=8, color='darkblue')
        axs[2, col].scatter(data, jitter, alpha=0.3, s=6, color='blue')
        axs[2, col].axvline(mean, color='red', linestyle='--', linewidth=2)
        axs[2, col].axvline(median, color='green', linestyle='--', linewidth=2)
        axs[2, col].set_ylim(-0.3, 0.3)
        axs[2, col].set_title(f'{label} - Number Line')
        axs[2, col].set_xlabel(label)
        axs[2, col].grid(True, alpha=0.3)

        try:
            density = stats.gaussian_kde(data)
            xs = np.linspace(min(data), max(data), 200)
            curve = density(xs)
            axs[3, col].plot(xs, curve, linewidth=2, color='purple')
            axs[3, col].fill_between(xs, curve, alpha=0.3, color='purple')
            axs[3, col].scatter(data, np.zeros_like(data), alpha=0.4, s=10, color='darkblue')
            axs[3, col].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
            axs[3, col].axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
            axs[3, col].legend()
            axs[3, col].set_title(f'{label} - Density Plot')
            axs[3, col].set_xlabel(label)
            axs[3, col].set_ylabel('Density')
            axs[3, col].grid(True, alpha=0.3)
        except Exception as e:
            axs[3, col].text(0.5, 0.5, f"KDE failed: {str(e)}", ha='center', va='center')
            axs[3, col].set_title(f'{label} - Density Plot')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined distribution plot saved to {save_path}")
    plt.show()

def plot_reactions(pos_reactions, neg_reactions, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    fig.suptitle('Positive and Negative Reaction Distributions', fontsize=16, fontweight='bold')

    # Positive reactions histogram
    axs[0].hist(pos_reactions, bins=30, color='green', alpha=0.7, edgecolor='black')
    axs[0].set_title('Positive Reactions per Post')
    axs[0].set_xlabel('Count')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True, alpha=0.3)
    axs[0].axvline(np.mean(pos_reactions), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pos_reactions):.2f}')
    axs[0].legend()

    # Negative reactions histogram
    axs[1].hist(neg_reactions, bins=30, color='red', alpha=0.7, edgecolor='black')
    axs[1].set_title('Negative Reactions per Post')
    axs[1].set_xlabel('Count')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True, alpha=0.3)
    axs[1].axvline(np.mean(neg_reactions), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(neg_reactions):.2f}')
    axs[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reactions distribution plot saved to {save_path}")
    plt.show()

def plot_cumulative_score(scores, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Clip scores at 1000 for cumulative plot only
    clipped_scores = [min(score, 1000) for score in scores]
    sorted_scores = np.sort(clipped_scores)[::-1]  # sort descending
    cumulative = np.cumsum(sorted_scores)
    cumulative_norm = cumulative / cumulative[-1]  # normalize to 1

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_norm, linewidth=2, color='blue')
    plt.title('Cumulative Sum of Scores (Clipped at 1000) Normalized to 1', fontsize=14, fontweight='bold')
    plt.xlabel('Posts sorted by score (descending)')
    plt.ylabel('Cumulative fraction of total score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cumulative score plot saved to {save_path}")
    plt.show()

def main():
    folder_path = "Cheif_Delphi_Jsons"

    print(f"Starting analysis of JSON files in '{folder_path}'...")
    print("=" * 50)

    scores = extract_field_from_jsons(folder_path, 'score')
    reader_counts = extract_field_from_jsons(folder_path, 'readers_count')
    trust_levels = extract_field_from_jsons(folder_path, 'trust_level')
    reaction_counts, pos_reactions, neg_reactions = extract_reactions_from_jsons(folder_path)

    print("=" * 50)
    print(f"Found {len(scores)} posts with scores.")
    print(f"Found {len(reader_counts)} posts with readers_count.")
    print(f"Found {len(trust_levels)} posts with trust_level.")

    if scores and reader_counts and trust_levels:
        print(f"SCORE - Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")
        print(f"READERS_COUNT - Mean: {np.mean(reader_counts):.2f}, Std: {np.std(reader_counts):.2f}")
        print(f"TRUST_LEVEL - Mean: {np.mean(trust_levels):.2f}, Std: {np.std(trust_levels):.2f}")

        print("=" * 50)
        print("Reaction Counts:")
        for k, v in reaction_counts.items():
            print(f"{k}: {v}")

        print("=" * 50)
        print(f"POSITIVE REACTIONS - Mean: {np.mean(pos_reactions):.2f}, Std: {np.std(pos_reactions):.2f}")
        print(f"NEGATIVE REACTIONS - Mean: {np.mean(neg_reactions):.2f}, Std: {np.std(neg_reactions):.2f}")

        plot_all_together_dual(scores, reader_counts, trust_levels, save_path="chief_delphi_combined_all.png")
        plot_reactions(pos_reactions, neg_reactions, save_path="chief_delphi_reactions.png")
        plot_cumulative_score(scores, save_path="chief_delphi_cumulative_score.png")

    else:
        print("Insufficient data found in JSON files.")

if __name__ == "__main__":
    main()
