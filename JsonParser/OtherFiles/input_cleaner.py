import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import os
import re
from difflib import SequenceMatcher

# Attempt to import BeautifulSoup; give helpful error if missing
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: bs4 (BeautifulSoup) is required. Install with: pip install beautifulsoup4")
    raise

DEBUG = True
MAC = True
SSH = False
LINUX = False
PRINT_MUCH = True

# # Similarity thresholds (tuneable)
# MIN_SENT_JACCARD = 0.15        # word-level Jaccard threshold for sentence match
# MIN_SENT_SEQ_RATIO = 0.55      # SequenceMatcher ratio threshold for sentence match
# MIN_LONGEST_CHAR_MATCH = 15    # minimum longest matching substring length (chars) to accept a sentence
# MIN_MODEL_KEEPED_RATIO = 0.15  # if final substituted text is < this fraction of baseline length, fallback

# # if MAC:
# #     MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# # else:
# #     MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# # if torch.backends.mps.is_available():
# #     DEVICE = "mps"
# #     TORCH_DTYPE = torch.float16
# # elif torch.cuda.is_available():
# #     DEVICE = "cuda"
# #     TORCH_DTYPE = torch.float16
# # else:
# #     DEVICE = "cpu"
# #     TORCH_DTYPE = torch.float16

# # print(f"Loading model {MODEL_NAME} once on {DEVICE} ...")
# # TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
# # MODEL = AutoModelForCausalLM.from_pretrained(
# #     MODEL_NAME,
# #     dtype=TORCH_DTYPE,
# #     trust_remote_code=True
# # ).to(DEVICE)
# # MODEL.eval()

def getTextFromLine(line):
    start = line.find("\"cooked\": \"")+len("'cooked': '")
    end = line.find("</p>\",")
    if DEBUG:
        print("START:", start)
        print("END:", end)
    return line[start:end]

def queryDeepSeek(input_text):
    if PRINT_MUCH:
        print(f"Starting DeepSeek query (input length: {len(input_text)} chars)...")

    messages = [{"role": "user", "content": input_text}]
    try:
        formatted_input = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted_input = f"User: {input_text}\nAssistant:"
        if PRINT_MUCH:
            print("Chat template failed, using fallback format")

    start_time = time.time()
    input_ids = TOKENIZER.encode(formatted_input, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10000,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=TOKENIZER.eos_token_id or TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id or TOKENIZER.pad_token_id,
        )

    response_ids = outputs[0][input_ids.shape[1]:]
    response = TOKENIZER.decode(response_ids, skip_special_tokens=True)

    end_time = time.time()
    if PRINT_MUCH:
        print(f"DeepSeek query completed in {end_time - start_time:.2f} seconds")

    return response.strip(), end_time - start_time

def clean_html(raw_html: str) -> str:
    """Algorithmically remove HTML tags, scripts, and styles, keeping text only, preserving paragraphs."""
    soup = BeautifulSoup(raw_html, "html.parser")

    for bad in soup(["script", "style"]):
        bad.decompose()

    text_parts = []
    for elem in soup.find_all(["p", "div", "li", "br"]):
        stripped = elem.get_text(" ", strip=True)
        if stripped:
            text_parts.append(stripped)

    if not text_parts:
        whole = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", whole).strip()

    return "\n\n".join(text_parts)

def split_into_sentences(text: str):
    """Naive sentence splitter - keeps sentences in order. Good enough for this task."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def word_jaccard(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.lower()))
    b_tokens = set(re.findall(r"\w+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

def longest_common_substring_len(a: str, b: str) -> int:
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return match.size

def extract_relevant_sentences(baseline: str, model_output: str, debug: bool=False):
    """
    Return the subset of sentences from model_output that appear to correspond to baseline,
    using multiple heuristics (word Jaccard, sequence ratio, longest-match).
    """
    baseline_norm = re.sub(r"\s+", " ", baseline).strip()
    model_norm = re.sub(r"\s+", " ", model_output).strip()

    model_sents = split_into_sentences(model_norm)
    kept_sents = []
    kept_info = []

    for s in model_sents:
        if not s:
            continue
        jacc = word_jaccard(s, baseline_norm)
        seq_ratio = SequenceMatcher(None, s, baseline_norm).ratio()
        lm = longest_common_substring_len(s, baseline_norm)

        keep = False
        if jacc >= MIN_SENT_JACCARD:
            keep = True
        if seq_ratio >= MIN_SENT_SEQ_RATIO:
            keep = True
        if lm >= MIN_LONGEST_CHAR_MATCH:
            keep = True

        if keep:
            kept_sents.append(s)
        if debug:
            kept_info.append({
                "sentence": s,
                "jaccard": jacc,
                "seq_ratio": seq_ratio,
                "longest_match": lm,
                "kept": keep
            })

    return kept_sents, kept_info

def merge_baseline_with_model(baseline: str, model_kept_sents: list, debug: bool=False):
    """
    For each baseline sentence, if the model has a corresponding kept sentence, prefer the model sentence (substitution).
    Otherwise keep the baseline sentence. Returns merged text.
    """
    base_sents = split_into_sentences(baseline)
    if not base_sents:
        # If baseline has no sentences (unlikely), just return concatenation of model_kept
        return " ".join(model_kept_sents).strip()

    merged = []
    used_model_indices = set()

    for bs in base_sents:
        best_idx = None
        best_score = 0.0
        for idx, ms in enumerate(model_kept_sents):
            if idx in used_model_indices:
                continue
            # score sentence pair by jaccard + seq ratio (weighted)
            jacc = word_jaccard(bs, ms)
            seq = SequenceMatcher(None, bs, ms).ratio()
            score = 0.6 * jacc + 0.4 * seq  # weighted
            if score > best_score:
                best_score = score
                best_idx = idx

        # Accept substitution if best_score passes a modest threshold
        if best_idx is not None and best_score >= 0.20:
            merged.append(model_kept_sents[best_idx])
            used_model_indices.add(best_idx)
            if debug:
                print(f"Substituting baseline sentence: '{bs[:60]}...' with model sentence: '{model_kept_sents[best_idx][:60]}...' score={best_score:.2f}")
        else:
            merged.append(bs)

    # If there are leftover model_kept_sents not used (e.g., extra lines model added that match something),
    # append them at the end if they add length (optional - safer to ignore)
    # final = join sentences with space, but attempt to preserve paragraph breaks from baseline by joining baseline_sents as they were
    return " ".join(merged).strip()

def strip_think_segment(text: str) -> str:
    """Strip any leading '</think>' and preceding content if present, otherwise return text unchanged."""
    idx = text.find("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].lstrip()
    return text

# ------------------ End helpers ------------------

def cleanText(data):
    posts = data["data"]["post_stream"]["posts"]

    if PRINT_MUCH:
        print(f"Processing {len(posts)} posts...")

    for i, post in enumerate(posts):
        if PRINT_MUCH:
            print(f"Processing post {i+1}/{len(posts)}")

        if "cooked" in post:
            string_org = post["cooked"]
            if DEBUG:
                print("Original text (raw HTML-like):", string_org[:200].replace("\n", " ") + ("..." if len(string_org) > 200 else ""))

            # Step 1: Algorithmic baseline cleaning (HTML -> plain text)
            baseline_clean = clean_html(string_org)
            if DEBUG:
                print("Baseline cleaned:", baseline_clean.replace("\n", " "))

            # # Step 2: Query DeepSeek with the already-cleaned version (not HTML)
            # prompt = (
            #     "Please rewrite/clean the following plain text so it reads naturally in standard English punctuation and grammar. "
            #     "IMPORTANT: Output ONLY the cleaned text. Do NOT add introductions, 'here is', explanations, or other commentary.\n\n"
            #     f"{baseline_clean}"
            # )
            # model_response, elapsed_time = queryDeepSeek(prompt)
            # model_output = strip_think_segment(model_response)

            # if DEBUG:
            #     print("Model raw output (first 200 chars):", model_output[:200].replace("\n", " ") + ("..." if len(model_output) > 200 else ""))

            # model_kept_sents, kept_info = extract_relevant_sentences(baseline_clean, model_output, debug=DEBUG)

            # if DEBUG:
            #     print("Kept sentences from model (count):", len(model_kept_sents))
            #     # Optionally print details
            #     for info in kept_info:
            #         if info["kept"]:
            #             print(f"KEPT: jaccard={info['jaccard']:.2f}, seq={info['seq_ratio']:.2f}, lm={info['longest_match']}, text='{info['sentence'][:120]}'")

            # if model_kept_sents:
            #     merged = merge_baseline_with_model(baseline_clean, model_kept_sents, debug=DEBUG)
            #     # If merged result is tiny (model didn't really contribute), fallback to baseline
            #     if len(merged) < max(10, int(len(baseline_clean) * MIN_MODEL_KEEPED_RATIO)):
            #         if DEBUG:
            #             print("Merged result too small; falling back to baseline_clean.")
            #         cleaned_text = baseline_clean
            #     else:
            #         cleaned_text = merged
            # else:
            #     if DEBUG:
            #         print("Model produced no relevant sentences. Using baseline_clean.")
            #     cleaned_text = baseline_clean

            # Assign the cleaned text to post["cooked"]
            post["cooked"] = baseline_clean

        #     if DEBUG:
        #         print()
        #         print("Final chosen text (first 300 chars):", cleaned_text[:300].replace("\n", " ") + ("..." if len(cleaned_text) > 300 else ""))
        #         print("Time taken:", elapsed_time, "seconds")
        # else:
        #     if PRINT_MUCH:
        #         print(f"Post {i+1} has no 'cooked' field, skipping")

    return data

def process_json_string(json_str):
    if PRINT_MUCH:
        print(f"Processing JSON string (length: {len(json_str)} chars)...")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("NOT VALID JSON AT ALL")
        print()
        print(f"Error: {e}")
        print()
        print()
        print()
        raise ValueError(f"Invalid JSON input: {e}")

    cleaned_data = cleanText(data)
    return json.dumps(cleaned_data, indent=4, ensure_ascii=False)

def main(args):
    filename = args.files[0]

    if PRINT_MUCH:
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE: {filename}")
        print(f"{'='*60}")

    # Fix the filename parsing - don't remove .json extension
    if filename.endswith('.json'):
        name = filename[:-5]  # Remove .json extension
        inputFileName = filename  # Use the original filename
    else:
        name = filename
        inputFileName = name + '.json'

    data = None
    content = None

    if PRINT_MUCH:
        print(f"Input file: {inputFileName}")

    try:
        if MAC or LINUX:
            with open(inputFileName, 'r') as inputFile:
                content = inputFile.read()
        else:
            with open(inputFileName, 'r', encoding='utf-8') as inputFile:
                content = inputFile.read()

        if PRINT_MUCH:
            print(f"File read successfully (length: {len(content)} chars)")

        try:
            data = json.loads(content)
            # Check if data is actually a string (JSON containing a string)
            if isinstance(data, str):
                if PRINT_MUCH:
                    print("Data is a JSON string, processing...")
                print("File contains JSON string, processing with process_json_string...")
                processed_content = process_json_string(data)
                data = json.loads(processed_content)
        except json.JSONDecodeError:
            if PRINT_MUCH:
                print("Initial JSON parsing failed, treating as raw string...")
            # If it's not valid JSON, treat it as a raw string and process it
            print("File contains raw string, processing with process_json_string...")
            processed_content = process_json_string(content)
            data = json.loads(processed_content)

        if isinstance(data["data"], str):
            if PRINT_MUCH:
                print("Parsing nested JSON in data['data']...")
            data["data"] = json.loads(data["data"])

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        if PRINT_MUCH:
            print(f"ERROR: File {inputFileName} not found")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        if PRINT_MUCH:
            print(f"ERROR reading {inputFileName}: {e}")
        return

    print()
    print()
    print()
    print()
    print("FILENAME:", inputFileName)
    print("TYPE:", type(data))
    print("TYPE data[data]:", type(data["data"]))
    print("TYPE data[data]:", type(data["data"]))

    data = cleanText(data)

    if SSH:
        outputFileName = "/home/rhayrapetyan/automatic/Cheif_Delphi_JSONS/" + name.split("/")[-1] + ".json"
    elif MAC:
        outputFileName = "/Users/rubenhayrapetyan/Downloads/Code/FRC/CheifDelphi-GPT/RAG_Model/Cheif_Delphi_JSONS/"+ name.split("/")[-1] + '.json'
    elif LINUX:
        outputFileName = "/mnt/c/Users/serge/Downloads/FRC/RAG_Model/Cheif_Delphi_JSONS/" + name.split("/")[-1] + '.json'
    else:
        file_name = os.path.splitext(os.path.basename(name))[0]  # just '150' without .json
        outputFileName = os.path.join(
            r"C:\Users\serge\Downloads\FRC\RAG_model\Cheif_Delphi_JSONS",
            f"{file_name}.json"
        )

    if PRINT_MUCH:
        print(f"Writing to: {outputFileName}")

    if DEBUG:
        print(f"Output written to {outputFileName}")
        print("Data = \n", data)

    if MAC or LINUX:
        with open(outputFileName, 'w') as outputFile:
            json.dump(data, outputFile, indent=4, ensure_ascii=False)
    else:
        with open(outputFileName, 'w', encoding='utf-8') as outputFile:
            json.dump(data, outputFile, indent=4, ensure_ascii=False)

    if PRINT_MUCH:
        print(f"COMPLETED: {filename}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON Parser")
    parser.add_argument('files', type=str, nargs='+', help='Input Clened JSON')
    args = parser.parse_args()
    main(args)