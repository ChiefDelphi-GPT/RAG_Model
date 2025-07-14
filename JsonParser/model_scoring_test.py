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
        "You will be given two texts. The first text will be a question/statement and the second text will be an response.\n"
        "Your job is to tell me how good of an response to the question/statement is the second text.\n"
        "You will generate a decimal score between 0.1 and 4.1 such that a score of 0.1 means that the second text is a terrible response to the question/statement presented in the first text and such that a score of 4.1 means that the second text is a terrific response to the question/statement presented in the first text.\n"
        "Think carefully\n"
        f"Here is the first text, the question/statement:\t{question}\n"
        f"Here is the second text, the response to the question/statement presented in the first text:\t{reply}"
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
# Main Processing
# -----------------------------

def process_folder(folder_path):
    results = []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as f:
            data = json.load(f)

        posts = data["data"]["post_stream"]["posts"]
        if len(posts) < 2:
            continue

        question = posts[0]["cooked"]
        for reply in tqdm(posts[1:], desc=f"Processing {file_name}"):
            reply_text = reply["cooked"]

            prompt = create_scoring_prompt(question, reply_text)
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


# -----------------------------
# Visualization & Print
# -----------------------------

def plot_scores(results, folder_prefix=""):
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


def print_all_comparisons(results):
    print("=" * 100)
    print("DETAILED COMPARISON OF MODEL VS ALGO SCORES")
    print("=" * 100)
    for idx, r in enumerate(results):
        print(f"\n[{idx+1}] File: {r['file']}")
        snippet = r['reply'][:100].replace('\n', ' ')
        print(f"Reply (snippet): {snippet}...")
        print(f"Model Score     : {r['model_score']:.2f}" if r['model_score'] is not None else "Model Score     : None")
        print(f"Algorithmic Score: {r['algo_score']:.2f}" if r['algo_score'] is not None else "Algorithmic Score: None")
        if r['model_score'] is not None and r['algo_score'] is not None:
            print(f"Difference       : {r['model_score'] - r['algo_score']:.2f}")


# -----------------------------
# Run Everything
# -----------------------------

if __name__ == "__main__":
    folder_path = input("Enter path to folder containing JSON files: ").strip()
    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path.")
        exit(1)

    print(f"\n📂 Parsing JSONs in: {folder_path}")
    results = process_folder(folder_path)

    if not results:
        print("No valid data found.")
    else:
        print_all_comparisons(results)
        plot_scores(results)

        with open("scoring_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("✅ Saved all results to scoring_results.json")
