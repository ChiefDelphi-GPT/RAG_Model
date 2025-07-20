import os
import json
import re
import time
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SCORE_CLIPPING = 1000
RECENCY_DECAY = 1080
Q_A_CLIPPING = 1500
MIN_TRUST_LEVEL = 0.1


# -----------------------------
# Helper functions
# -----------------------------

def diff_days(date_obj):
    today = dt.date.today()
    return (today - date_obj).days


def extract_float_in_range(text, low=0.1, high=4.1):
    numbers = re.findall(r"-?\d+\.?\d*", text)
    valid = [float(n) for n in numbers if low <= float(n) <= high]
    return valid[-1] if valid else None


def create_scoring_prompt(question, reply):
    return (
        "You will receive two texts: a question or statement as the first text, and a response as the second text. Your task is to evaluate how well the second text responds to the first. Provide a decimal score between 0.1 and 4.1, where 0.1 indicates an unhelpful response and 4.1 indicates an excellent response. Your score should reflect the helpfulness or quality of the response in relation to the question or statement.\n",
        "Output only the score\n",
        f"Here is the question/statement:\t{question}\n",
        f"Here is the response to the question/statement:\t{reply}"
    )


def query_deepseek(prompt):
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True
    ).to(device)

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        formatted_input = f"User: {prompt}\nAssistant:"

    input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"MODEL RESPONSE: {response}")
    score = extract_float_in_range(response)
    print(f"[AI Score Computed] => {score}")
    return score


def compute_algorithmic_score(reply):
    try:
        created_at = reply["created_at"]
        date_parts = [int(x) for x in created_at.split("T")[0].split("-")]
        date_obj = dt.date(*date_parts)
        difference = diff_days(date_obj)
        recency_score = np.exp(-1.0 * difference / RECENCY_DECAY)
        confidence_score = sqrt(reply.get("readers_count", 1))
        trust_level = reply.get("trust_level", 0)
        raw_score = reply.get("score", 0)
        clipped_score = min(raw_score, SCORE_CLIPPING)

        final_score = sqrt(min(recency_score * confidence_score * sqrt(clipped_score) * (trust_level + MIN_TRUST_LEVEL), Q_A_CLIPPING))
        print(f"[Algo Score Computed] => {final_score:.2f}")
        return final_score
    except Exception as e:
        print(f"Error computing algo score: {e}")
        return None


# -----------------------------
# Statistical Analysis Functions
# -----------------------------

def compute_descriptive_stats(data, name):
    """Compute comprehensive descriptive statistics"""
    clean_data = [x for x in data if x is not None]
    if not clean_data:
        return {}
    
    stats_dict = {
        'count': len(clean_data),
        'mean': np.mean(clean_data),
        'median': np.median(clean_data),
        'std': np.std(clean_data),
        'var': np.var(clean_data),
        'min': np.min(clean_data),
        'max': np.max(clean_data),
        'range': np.max(clean_data) - np.min(clean_data),
        'q1': np.percentile(clean_data, 25),
        'q3': np.percentile(clean_data, 75),
        'iqr': np.percentile(clean_data, 75) - np.percentile(clean_data, 25),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data),
        'cv': np.std(clean_data) / np.mean(clean_data) if np.mean(clean_data) != 0 else 0
    }
    
    return stats_dict


def print_statistical_summary(results):
    """Print comprehensive statistical analysis"""
    model_scores = [r["model_score"] for r in results if r["model_score"] is not None]
    algo_scores = [r["algo_score"] for r in results if r["algo_score"] is not None]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*80)
    
    # Individual distribution analysis
    model_stats = compute_descriptive_stats(model_scores, "Model")
    algo_stats = compute_descriptive_stats(algo_scores, "Algorithmic")
    
    print("\n📊 DESCRIPTIVE STATISTICS")
    print("-" * 50)
    print(f"{'Metric':<15} {'Model Score':<15} {'Algo Score':<15} {'Difference':<15}")
    print("-" * 60)
    
    for key in ['count', 'mean', 'median', 'std', 'var', 'min', 'max', 'range', 'q1', 'q3', 'iqr', 'skewness', 'kurtosis', 'cv']:
        model_val = model_stats.get(key, 0)
        algo_val = algo_stats.get(key, 0)
        diff = model_val - algo_val if isinstance(model_val, (int, float)) and isinstance(algo_val, (int, float)) else 'N/A'
        print(f"{key:<15} {model_val:<15.3f} {algo_val:<15.3f} {diff}")
    
    # Correlation analysis
    if len(model_scores) > 1 and len(algo_scores) > 1:
        # Align scores (only use pairs where both exist)
        paired_data = [(m, a) for m, a in zip([r["model_score"] for r in results], [r["algo_score"] for r in results]) 
                       if m is not None and a is not None]
        
        if len(paired_data) > 1:
            model_paired = [x[0] for x in paired_data]
            algo_paired = [x[1] for x in paired_data]
            
            pearson_r, pearson_p = stats.pearsonr(model_paired, algo_paired)
            spearman_r, spearman_p = stats.spearmanr(model_paired, algo_paired)
            
            print(f"\n🔗 CORRELATION ANALYSIS")
            print("-" * 50)
            print(f"Pearson correlation:  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
            print(f"Spearman correlation: r = {spearman_r:.4f}, p = {spearman_p:.4f}")
            
            # Regression metrics
            mse = mean_squared_error(algo_paired, model_paired)
            mae = mean_absolute_error(algo_paired, model_paired)
            r2 = r2_score(algo_paired, model_paired)
            
            print(f"\n📈 REGRESSION METRICS (Model vs Algo)")
            print("-" * 50)
            print(f"Mean Squared Error:   {mse:.4f}")
            print(f"Mean Absolute Error:  {mae:.4f}")
            print(f"R² Score:             {r2:.4f}")
    
    # Distribution tests
    print(f"\n📊 NORMALITY TESTS")
    print("-" * 50)
    
    if len(model_scores) > 8:
        shapiro_model = stats.shapiro(model_scores)
        print(f"Model Scores - Shapiro-Wilk: W = {shapiro_model.statistic:.4f}, p = {shapiro_model.pvalue:.4f}")
    
    if len(algo_scores) > 8:
        shapiro_algo = stats.shapiro(algo_scores)
        print(f"Algo Scores - Shapiro-Wilk:  W = {shapiro_algo.statistic:.4f}, p = {shapiro_algo.pvalue:.4f}")
    
    # Comparing distributions
    if len(model_scores) > 1 and len(algo_scores) > 1:
        print(f"\n⚖️ DISTRIBUTION COMPARISON TESTS")
        print("-" * 50)
        
        # Mann-Whitney U test
        mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(model_scores, algo_scores, alternative='two-sided')
        print(f"Mann-Whitney U test: U = {mannwhitney_stat:.4f}, p = {mannwhitney_p:.4f}")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(model_scores, algo_scores)
        print(f"Kolmogorov-Smirnov:  D = {ks_stat:.4f}, p = {ks_p:.4f}")
        
        # If paired data available
        if len(paired_data) > 1:
            differences = [m - a for m, a in paired_data]
            if len(differences) > 1:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(differences)
                print(f"Wilcoxon signed-rank: W = {wilcoxon_stat:.4f}, p = {wilcoxon_p:.4f}")


def create_comprehensive_plots(results, folder_prefix=""):
    """Create comprehensive statistical visualizations"""
    model_scores = [r["model_score"] for r in results if r["model_score"] is not None]
    algo_scores = [r["algo_score"] for r in results if r["algo_score"] is not None]
    
    # Create paired data for correlation plots
    paired_data = [(m, a) for m, a in zip([r["model_score"] for r in results], [r["algo_score"] for r in results]) 
                   if m is not None and a is not None]
    model_paired = [x[0] for x in paired_data]
    algo_paired = [x[1] for x in paired_data]
    differences = [m - a for m, a in paired_data]
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Distribution comparison (histograms)
    ax1 = plt.subplot(4, 3, 1)
    plt.hist(model_scores, alpha=0.7, bins=20, label='Model Scores', color='skyblue', edgecolor='black')
    plt.hist(algo_scores, alpha=0.7, bins=20, label='Algo Scores', color='lightcoral', edgecolor='black')
    plt.xlabel('Score Value')
    plt.ylabel('Frequency')
    plt.title('Score Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Box plots side by side
    ax2 = plt.subplot(4, 3, 2)
    data_to_plot = [model_scores, algo_scores]
    box_plot = plt.boxplot(data_to_plot, labels=['Model', 'Algorithmic'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('skyblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    plt.title('Score Distribution (Box Plot)')
    plt.ylabel('Score Value')
    plt.grid(True, alpha=0.3)
    
    # 3. Violin plots
    ax3 = plt.subplot(4, 3, 3)
    violin_data = [model_scores, algo_scores]
    violin_plot = plt.violinplot(violin_data, positions=[1, 2], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ['Model', 'Algorithmic'])
    plt.title('Score Distribution (Violin Plot)')
    plt.ylabel('Score Value')
    plt.grid(True, alpha=0.3)
    
    # 4. Scatter plot with regression line
    ax4 = plt.subplot(4, 3, 4)
    if len(model_paired) > 1:
        plt.scatter(algo_paired, model_paired, alpha=0.6, s=50)
        # Add regression line
        z = np.polyfit(algo_paired, model_paired, 1)
        p = np.poly1d(z)
        plt.plot(algo_paired, p(algo_paired), "r--", alpha=0.8, linewidth=2)
        # Add diagonal line
        min_val = min(min(algo_paired), min(model_paired))
        max_val = max(max(algo_paired), max(model_paired))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
    plt.xlabel('Algorithmic Score')
    plt.ylabel('Model Score')
    plt.title('Correlation: Model vs Algorithmic Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 5. Difference analysis
    ax5 = plt.subplot(4, 3, 5)
    if differences:
        plt.hist(differences, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
        plt.xlabel('Model Score - Algorithmic Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Score Differences')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. Cumulative distribution functions
    ax6 = plt.subplot(4, 3, 6)
    if model_scores and algo_scores:
        sorted_model = np.sort(model_scores)
        sorted_algo = np.sort(algo_scores)
        y_model = np.arange(1, len(sorted_model) + 1) / len(sorted_model)
        y_algo = np.arange(1, len(sorted_algo) + 1) / len(sorted_algo)
        plt.plot(sorted_model, y_model, label='Model Scores', linewidth=2)
        plt.plot(sorted_algo, y_algo, label='Algorithmic Scores', linewidth=2)
        plt.xlabel('Score Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 7. Q-Q plot
    ax7 = plt.subplot(4, 3, 7)
    if len(model_scores) > 1 and len(algo_scores) > 1:
        stats.probplot(model_scores, dist="norm", plot=plt)
        plt.title('Q-Q Plot: Model Scores vs Normal Distribution')
        plt.grid(True, alpha=0.3)
    
    # 8. Q-Q plot for algorithmic scores
    ax8 = plt.subplot(4, 3, 8)
    if len(algo_scores) > 1:
        stats.probplot(algo_scores, dist="norm", plot=plt)
        plt.title('Q-Q Plot: Algo Scores vs Normal Distribution')
        plt.grid(True, alpha=0.3)
    
    # 9. Residual plot
    ax9 = plt.subplot(4, 3, 9)
    if len(model_paired) > 1 and len(algo_paired) > 1:
        # Calculate residuals
        z = np.polyfit(algo_paired, model_paired, 1)
        p = np.poly1d(z)
        residuals = np.array(model_paired) - p(np.array(algo_paired))
        plt.scatter(algo_paired, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Algorithmic Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
    
    # 10. Score ranges comparison
    ax10 = plt.subplot(4, 3, 10)
    if model_scores and algo_scores:
        model_stats = compute_descriptive_stats(model_scores, "Model")
        algo_stats = compute_descriptive_stats(algo_scores, "Algo")
        
        metrics = ['mean', 'median', 'std', 'min', 'max']
        model_vals = [model_stats.get(m, 0) for m in metrics]
        algo_vals = [algo_stats.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, model_vals, width, label='Model', color='skyblue', edgecolor='black')
        plt.bar(x + width/2, algo_vals, width, label='Algorithmic', color='lightcoral', edgecolor='black')
        
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Statistical Metrics Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 11. Percentile comparison
    ax11 = plt.subplot(4, 3, 11)
    if model_scores and algo_scores:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        model_percentiles = [np.percentile(model_scores, p) for p in percentiles]
        algo_percentiles = [np.percentile(algo_scores, p) for p in percentiles]
        
        plt.plot(percentiles, model_percentiles, 'o-', label='Model', linewidth=2, markersize=8)
        plt.plot(percentiles, algo_percentiles, 's-', label='Algorithmic', linewidth=2, markersize=8)
        plt.xlabel('Percentile')
        plt.ylabel('Score Value')
        plt.title('Percentile Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 12. Bland-Altman plot
    ax12 = plt.subplot(4, 3, 12)
    if len(model_paired) > 1 and len(algo_paired) > 1:
        means = [(m + a) / 2 for m, a in zip(model_paired, algo_paired)]
        diffs = [m - a for m, a in zip(model_paired, algo_paired)]
        
        plt.scatter(means, diffs, alpha=0.6)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        plt.axhline(y=mean_diff, color='red', linestyle='-', label=f'Mean diff: {mean_diff:.3f}')
        plt.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--', alpha=0.7, label=f'+1.96σ: {mean_diff + 1.96*std_diff:.3f}')
        plt.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--', alpha=0.7, label=f'-1.96σ: {mean_diff - 1.96*std_diff:.3f}')
        
        plt.xlabel('Mean of Two Scores')
        plt.ylabel('Difference (Model - Algo)')
        plt.title('Bland-Altman Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{folder_prefix}comprehensive_statistical_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed plots
    create_distribution_analysis_plots(model_scores, algo_scores, folder_prefix)


def create_distribution_analysis_plots(model_scores, algo_scores, folder_prefix=""):
    """Create detailed distribution analysis plots"""
    
    # Distribution shape analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Histogram with kernel density estimation
    axes[0, 0].hist(model_scores, bins=20, alpha=0.7, density=True, label='Model Scores', color='skyblue', edgecolor='black')
    axes[0, 0].hist(algo_scores, bins=20, alpha=0.7, density=True, label='Algo Scores', color='lightcoral', edgecolor='black')
    
    # Add KDE
    if len(model_scores) > 1:
        kde_model = stats.gaussian_kde(model_scores)
        x_range = np.linspace(min(model_scores), max(model_scores), 100)
        axes[0, 0].plot(x_range, kde_model(x_range), 'b-', linewidth=2, label='Model KDE')
    
    if len(algo_scores) > 1:
        kde_algo = stats.gaussian_kde(algo_scores)
        x_range_algo = np.linspace(min(algo_scores), max(algo_scores), 100)
        axes[0, 0].plot(x_range_algo, kde_algo(x_range_algo), 'r-', linewidth=2, label='Algo KDE')
    
    axes[0, 0].set_title('Histogram with Kernel Density Estimation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plots with outliers marked
    box_data = [model_scores, algo_scores]
    bp = axes[0, 1].boxplot(box_data, labels=['Model', 'Algorithmic'], patch_artist=True, 
                           showfliers=True, flierprops=dict(marker='o', markersize=5, alpha=0.7))
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[0, 1].set_title('Box Plot with Outliers')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Empirical CDF comparison
    if model_scores and algo_scores:
        model_sorted = np.sort(model_scores)
        algo_sorted = np.sort(algo_scores)
        model_cdf = np.arange(1, len(model_sorted) + 1) / len(model_sorted)
        algo_cdf = np.arange(1, len(algo_sorted) + 1) / len(algo_sorted)
        
        axes[0, 2].step(model_sorted, model_cdf, where='post', label='Model CDF', linewidth=2)
        axes[0, 2].step(algo_sorted, algo_cdf, where='post', label='Algo CDF', linewidth=2)
        axes[0, 2].set_xlabel('Score Value')
        axes[0, 2].set_ylabel('Cumulative Probability')
        axes[0, 2].set_title('Empirical Cumulative Distribution Functions')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Probability density comparison
    if len(model_scores) > 1 and len(algo_scores) > 1:
        axes[1, 0].hist(model_scores, bins=20, alpha=0.7, density=True, label='Model', color='skyblue', edgecolor='black')
        axes[1, 0].hist(algo_scores, bins=20, alpha=0.7, density=True, label='Algo', color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Score Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Probability Density Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Quantile-Quantile plot between the two distributions
    if len(model_scores) > 1 and len(algo_scores) > 1:
        model_quantiles = np.percentile(model_scores, np.linspace(0, 100, 101))
        algo_quantiles = np.percentile(algo_scores, np.linspace(0, 100, 101))
        
        axes[1, 1].scatter(algo_quantiles, model_quantiles, alpha=0.7, s=30)
        # Add diagonal line
        min_val = min(min(model_quantiles), min(algo_quantiles))
        max_val = max(max(model_quantiles), max(algo_quantiles))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect agreement')
        axes[1, 1].set_xlabel('Algorithmic Score Quantiles')
        axes[1, 1].set_ylabel('Model Score Quantiles')
        axes[1, 1].set_title('Q-Q Plot: Model vs Algorithmic Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Distribution statistics radar chart
    if model_scores and algo_scores:
        model_stats = compute_descriptive_stats(model_scores, "Model")
        algo_stats = compute_descriptive_stats(algo_scores, "Algo")
        
        # Normalize statistics for radar chart
        metrics = ['mean', 'std', 'skewness', 'kurtosis']
        model_vals = [model_stats.get(m, 0) for m in metrics]
        algo_vals = [algo_stats.get(m, 0) for m in metrics]
        
        # Simple bar chart instead of radar for clarity
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, model_vals, width, label='Model', color='skyblue', edgecolor='black')
        axes[1, 2].bar(x + width/2, algo_vals, width, label='Algorithmic', color='lightcoral', edgecolor='black')
        axes[1, 2].set_xlabel('Statistical Measures')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_title('Distribution Shape Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{folder_prefix}detailed_distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


# -----------------------------
# Original functions (modified)
# -----------------------------

def process_folder(folder_path):
    results = []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load JSON from {file_name}: {e}")
            continue

        # Confirm it's a dict
        if not isinstance(data, dict):
            print(f"⚠️ Skipping {file_name}: not a JSON object (type={type(data)})")
            continue

        try:
            posts = data["data"]["post_stream"]["posts"]
        except Exception as e:
            print(f"❌ Structure error in {file_name}: {e}")
            print(f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            continue

        if len(posts) < 2:
            continue

        question = posts[0]["cooked"]
        for reply in tqdm(posts[1:], desc=f"Processing {file_name}"):
            reply_text = reply["cooked"]

            prompt = create_scoring_prompt(question, reply_text)
            print(f"QUESTION: {question}")
            print(f"RESPONSE: {reply_text}")
            model_score = query_deepseek(prompt)
            algo_score = compute_algorithmic_score(reply)

            results.append({
                "file": file_name,
                "question": question,
                "reply": reply_text,
                "model_score": model_score,
                "algo_score": algo_score
            })

    return results



def plot_scores(results, folder_prefix=""):
    """Original plotting function (kept for backward compatibility)"""
    model_scores = [r["model_score"] for r in results]
    algo_scores = [r["algo_score"] for r in results]
    diffs = [m - a if m is not None and a is not None else 0 for m, a in zip(model_scores, algo_scores)]
    indices = np.arange(len(results))

    # 1. Line plot
    plt.figure(figsize=(12, 5))
    plt.plot(indices, model_scores, label="Model Score", marker='o')
    plt.plot(indices, algo_scores, label="Algorithmic Score", marker='x')
    plt.xlabel("Reply Index")
    plt.ylabel("Score")
    plt.title("Model vs Algorithmic Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder_prefix}comparison_line_plot.png")
    plt.show()

    # 2. Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(algo_scores, model_scores, alpha=0.7)
    plt.plot([0, 10], [0, 10], '--', color='gray')
    plt.xlabel("Algorithmic Score")
    plt.ylabel("Model Score")
    plt.title("Scatter: Model vs Algorithmic")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder_prefix}scatter_plot.png")
    plt.show()

    # 3. Histogram of differences
    plt.figure(figsize=(10, 5))
    plt.hist(diffs, bins=20, edgecolor='black')
    plt.xlabel("Model Score - Algorithmic Score")
    plt.ylabel("Count")
    plt.title("Distribution of Score Differences")
    plt.tight_layout()
    plt.savefig(f"{folder_prefix}score_difference_histogram.png")
    plt.show()

    # 4. Boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot([model_scores, algo_scores], labels=['Model', 'Algorithmic'])
    plt.title("Score Distribution (Boxplot)")
    plt.tight_layout()
    plt.savefig(f"{folder_prefix}boxplot_scores.png")
    plt.show()

    # 5. Bar chart of deltas
    plt.figure(figsize=(14, 5))
    plt.bar(indices, diffs)
    plt.title("Score Differences (Model - Algorithmic)")
    plt.xlabel("Reply Index")
    plt.ylabel("Difference")
    plt.tight_layout()
    plt.savefig(f"{folder_prefix}delta_barplot.png")
    plt.show()

    print("\n✅ Saved all plots.\n")

def main():
    # Example folder path, adjust as needed
    folder_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/Cheif_Delphi_JSONS"
    
    # Process the folder and obtain results
    results = process_folder(folder_path)
    
    # Print statistical summary
    print_statistical_summary(results)
    
    # Create comprehensive plots
    create_comprehensive_plots(results)

if __name__ == "__main__":
    main()