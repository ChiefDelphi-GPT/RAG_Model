import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_scores_from_jsons(folder_path):
    """
    Extract all scores from JSON files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
    
    Returns:
        list: List of all scores found in the JSON files
    """
    scores = []
    folder = Path(folder_path)
    
    # Loop through JSON files 0.json to 149.json
    for i in range(150):
        json_file = folder / f"{i}.json"
        
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Navigate to posts array
                if 'data' in data and 'post_stream' in data['data'] and 'posts' in data['data']['post_stream']:
                    posts = data['data']['post_stream']['posts']
                    
                    # Extract score from each post
                    for post in posts:
                        if 'score' in post and post['score'] is not None:
                            scores.append(post['score'])
                
                print(f"Processed {json_file.name} - Found {len([p for p in data['data']['post_stream']['posts'] if 'score' in p])} posts with scores")
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {json_file.name}: {e}")
            except KeyError as e:
                print(f"Missing expected key in {json_file.name}: {e}")
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
        else:
            print(f"File {json_file.name} not found")
    
    return scores

def create_scatterplot(scores, save_path=None):
    """
    Create a scatterplot of the scores.
    
    Args:
        scores (list): List of scores to plot
        save_path (str, optional): Path to save the plot
    """
    if not scores:
        print("No scores found to plot!")
        return
    
    # Create index for x-axis (post number/order)
    x_values = list(range(len(scores)))
    
    # Create the scatterplot
    plt.figure(figsize=(12, 8))
    plt.scatter(x_values, scores, alpha=0.6, s=20)
    
    # Customize the plot
    plt.title(f'Chief Delphi Post Scores Distribution\n(Total Posts: {len(scores)})', fontsize=14, fontweight='bold')
    plt.xlabel('Post Index', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    max_score = max(scores)
    min_score = min(scores)
    
    stats_text = f'Mean: {mean_score:.2f}\nMedian: {median_score:.2f}\nMax: {max_score:.2f}\nMin: {min_score:.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.show()

def create_histogram(scores, save_path=None):
    """
    Create a histogram of the scores for additional analysis.
    
    Args:
        scores (list): List of scores to plot
        save_path (str, optional): Path to save the plot
    """
    if not scores:
        print("No scores found to plot!")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    plt.title(f'Chief Delphi Post Scores Distribution (Histogram)\n(Total Posts: {len(scores)})', fontsize=14, fontweight='bold')
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    
    plt.show()

def create_number_line_distribution(scores, save_path=None):
    """
    Create a number line distribution showing scores density along a horizontal line.
    
    Args:
        scores (list): List of scores to plot
        save_path (str, optional): Path to save the plot
    """
    if not scores:
        print("No scores found to plot!")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top plot: Traditional histogram for context
    ax1.hist(scores, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title(f'Chief Delphi Post Scores Distribution\n(Total Posts: {len(scores)})', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics to top plot
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    ax1.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
    ax1.legend()
    
    # Bottom plot: Number line distribution
    # Create a rug plot (marks on the x-axis) for each score
    ax2.scatter(scores, np.zeros_like(scores), alpha=0.6, s=8, color='darkblue')
    
    # Add jitter to y-axis to spread out overlapping points
    jitter = np.random.normal(0, 0.05, len(scores))
    ax2.scatter(scores, jitter, alpha=0.3, s=6, color='blue')
    
    # Customize the number line plot
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_ylabel('Distribution', fontsize=12)
    ax2.set_title('Number Line Distribution (Each dot represents a post)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.3, 0.3)
    
    # Add statistical markers on the number line
    ax2.axvline(mean_score, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(median_score, color='green', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add quartile markers
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    ax2.axvline(q1, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q1: {q1:.2f}')
    ax2.axvline(q3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q3: {q3:.2f}')
    
    # Add text annotations
    ax2.text(mean_score, 0.25, f'Mean\n{mean_score:.2f}', ha='center', va='bottom', 
             fontsize=9, color='red', fontweight='bold')
    ax2.text(median_score, -0.25, f'Median\n{median_score:.2f}', ha='center', va='top', 
             fontsize=9, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Number line distribution saved to {save_path}")
    
    plt.show()

def create_density_plot(scores, save_path=None):
    """
    Create a kernel density estimation plot showing smooth distribution curve.
    
    Args:
        scores (list): List of scores to plot
        save_path (str, optional): Path to save the plot
    """
    if not scores:
        print("No scores found to plot!")
        return
    
    try:
        from scipy import stats
        
        plt.figure(figsize=(12, 6))
        
        # Create density plot
        density = stats.gaussian_kde(scores)
        xs = np.linspace(min(scores), max(scores), 200)
        density_curve = density(xs)
        
        plt.plot(xs, density_curve, linewidth=2, color='purple', label='Density Curve')
        plt.fill_between(xs, density_curve, alpha=0.3, color='purple')
        
        # Add individual score markers on x-axis
        plt.scatter(scores, np.zeros_like(scores), alpha=0.4, s=10, color='darkblue', 
                   label=f'Individual Scores (n={len(scores)})')
        
        # Add statistical lines
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
        
        plt.title('Chief Delphi Post Scores - Density Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Density plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        print("SciPy not available. Skipping density plot. Install with: pip install scipy")

def main():
    """
    Main function to run the analysis.
    """
    # Set the folder path
    folder_path = "Cheif_Delphi_Jsons"  # Note: keeping the original spelling as specified
    
    print(f"Starting analysis of JSON files in '{folder_path}'...")
    print("=" * 50)
    
    # Extract scores from all JSON files
    scores = extract_scores_from_jsons(folder_path)
    
    print("=" * 50)
    print(f"Analysis complete! Found {len(scores)} posts with scores.")
    
    if scores:
        print(f"Score range: {min(scores):.2f} to {max(scores):.2f}")
        print(f"Average score: {np.mean(scores):.2f}")
        print(f"Median score: {np.median(scores):.2f}")
        
        # Create scatterplot
        create_scatterplot(scores, save_path="chief_delphi_scores_scatter.png")
        
        # Create histogram for additional insight
        create_histogram(scores, save_path="chief_delphi_scores_histogram.png")
        
        # Create number line distribution
        create_number_line_distribution(scores, save_path="chief_delphi_scores_numberline.png")
        
        # Create density plot
        create_density_plot(scores, save_path="chief_delphi_scores_density.png")
    else:
        print("No scores found in the JSON files. Please check the file structure and paths.")

if __name__ == "__main__":
    main()