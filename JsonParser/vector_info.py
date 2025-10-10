import argparse
import json
import datetime as dt
import numpy as np
from math import sqrt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import uuid
import re
from html import unescape

MAC = False
LINUX = False
DEBUG = True
SCORE_CLIPPING = 1000
RECENCY_DECAY = 1080
MIN_TRUST_LEVEL = 0.1
Q_A_CLIPPINNG = 1500
question = None
vectors = []

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def setup_device_and_model_cpu():
    """Fallback CPU setup"""
    device = "cpu"
    torch_dtype = torch.float32
    print(f"Loading model {MODEL_NAME} on CPU...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    return device, torch_dtype, tokenizer, model


def setup_device_and_model():
    """Setup device and load model with proper error handling"""
    
    # Device selection with more detailed info
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
        print(f"Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using NVIDIA GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"Available GPUs: {gpu_count}")
        
        # Optionally specify which GPU to use (0 is default)
        device = f"cuda:0"  # or just "cuda" for default
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("Using CPU (no GPU detected)")
    
    print(f"Loading model {MODEL_NAME} on {device}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model with error handling
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if device.startswith("cuda") else None  # Automatic GPU mapping
        )
        
        # Move to device if not using device_map
        if not device.startswith("cuda"):
            model = model.to(device)
        
        model.eval()
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        print(f"Model successfully loaded on: {model_device}")
        
        # Test GPU memory usage if using CUDA
        if device.startswith("cuda"):
            torch.cuda.empty_cache()  # Clear cache
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_cached = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Cached: {memory_cached:.2f} GB")
        
        return device, torch_dtype, tokenizer, model
        
    except Exception as e:
        print(f"Error loading model on {device}: {e}")
        if device != "cpu":
            print("Falling back to CPU...")
            return setup_device_and_model_cpu()
        else:
            raise e
        
DEVICE, TORCH_DTYPE, TOKENIZER, MODEL = setup_device_and_model()
print("Loading embedding model on CPU...")
EMBEDDING_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")

def getTextFromLine(line):
    start = line.find("\"cooked\": \"")+len("'cooked': '")
    end = line.find("</p>\",")
    if DEBUG:
        print("START:", start)
        print("END:", end)
    return line[start:end]

def diff_days(other_date, to_date=dt.datetime.today().date()):
    return (to_date - other_date).days

def extractFeatures(data):
    global vectors, question
    posts = data["data"]["post_stream"]["posts"]
    q_a = None
    replies = []
    for i, post in enumerate(posts):
        if (i == 0): #evaluate the question
            print()
            print("DOING QUESTION")
            post["score"] = min(post["score"], SCORE_CLIPPING)                
            difference = diff_days(dt.date(int(post["created_at"].split("T")[0].split("-")[0]),
                int(post["created_at"].split("T")[0].split("-")[1]),
                int(post["created_at"].split("T")[0].split("-")[2])))
            recencyScore = np.exp(-1.0 * (difference) / RECENCY_DECAY)
            confidenceScore = sqrt(post["readers_count"])
            q_a = ([post["cooked"], post["topic_slug"]], 
                   sqrt(min(recencyScore * confidenceScore * sqrt(post["score"]) * (post["trust_level"]+MIN_TRUST_LEVEL), Q_A_CLIPPINNG)), 
                   post["id"])
            if DEBUG:
                print(f"QA: {q_a}")
        else:
            replies.append(post)
    print()
    print("DOING REPLIES")
    reply_data, best_replies = scoreReplies(replies)
    replies = pick_best_replies(reply_data, best_replies)
    if DEBUG:
        print()
        print()
        print()
        print("QUESTIONS")
        print(q_a)
        print()
        print()
        print("REPLIES")
        print(replies)
    return (q_a, replies)

def strip_html(raw):
    # remove HTML tags
    no_tags = re.sub(r"<[^>]+>", "", raw)
    return unescape(no_tags).strip()


def queryDeepSeek(input_text):
    if DEBUG:
        print(f"Starting DeepSeek query (input length: {len(input_text)} chars)...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    messages = [{"role": "user", "content": input_text}]
    try:
        formatted_input = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted_input = f"User: {input_text}\nAssistant:"

    start_time = time.time()
    input_ids = TOKENIZER.encode(formatted_input, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=TOKENIZER.eos_token_id or TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id or TOKENIZER.pad_token_id,
        )

    response_ids = outputs[0][input_ids.shape[1]:]
    response = TOKENIZER.decode(response_ids, skip_special_tokens=True)

    del input_ids, attention_mask, outputs, response_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    end_time = time.time()
    if DEBUG:
        print(f"DeepSeek query completed in {end_time - start_time:.2f} seconds")

    return response.strip(), end_time - start_time


def extract_float_in_range(text):
    """
    Extract the first floating point number between 0.1 and 4.1 from a string.
    Handles + sign, scientific notation, and thousands separators.
    """
    # Normalize
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    
    pattern = r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?'
    matches = re.findall(pattern, text)

    for match in matches:
        try:
            num = float(match)
            if 0.1 <= num <= 4.1:
                return num
            elif num > 4.1:
                return 4.1
            elif num < 0.1:
                return 0.1
        except ValueError:
            continue
    return None

def scoreReplies(replies):
    replies_data = []
    best_reps = []
    for reply in replies:
        if reply["accepted_answer"] or reply["topic_accepted_answer"]:
            if DEBUG:
                print("A reply that was the accepted answer:", reply["cooked"])
            best_reps.append(reply)
        reply["score"] = min(reply["score"], SCORE_CLIPPING)                
        difference = diff_days(dt.date(int(reply["created_at"].split("T")[0].split("-")[0]),
            int(reply["created_at"].split("T")[0].split("-")[1]),
            int(reply["created_at"].split("T")[0].split("-")[2])))
        recencyScore = np.exp(-1.0 * (difference) / RECENCY_DECAY)
        confidenceScore = sqrt(reply["readers_count"])

        reply_score = sqrt(min(recencyScore * confidenceScore * sqrt(reply["score"]) * (reply["trust_level"]+MIN_TRUST_LEVEL), Q_A_CLIPPINNG))
        prompt = (
            "You will receive two texts: a question or statement as the first text, and a response as the second text. Your task is to evaluate how well the second text responds to the first. Provide a decimal score between 0.1 and 4.1, where 0.1 indicates an unhelpful response and 4.1 indicates an excellent response. Your score should reflect the helpfulness or quality of the response in relation to the question or statement.\n",
            "Output only the score\n",
            f"Here is the question/statement:\t{question}\n",
            f'Here is the response to the question/statement:\t{reply["cooked"]}'
        )
        model_response, elapsed_time = queryDeepSeek(prompt)
        ai_score = extract_float_in_range(model_response)
        if DEBUG:
            print("AI Score:", ai_score)
        if (ai_score):
            score = reply_score * float(ai_score)
            if DEBUG:
                print()
                print()
                print()
                print("Model response:", model_response)
                print("Time taken:", elapsed_time, "seconds")
                print("Score:", score)
                print("The reply:", reply["cooked"])
            reply_touple = (reply["cooked"], score)
            replies_data.append(reply_touple)
        else:
            continue
    return replies_data, best_reps

def pick_best_replies(replies_data, best_replies):
    """
    Pick exactly 5 best replies:
    - Always include accepted answers (best_replies).
    - Fill the rest with the top scoring replies from replies_data.
    - Ensure no duplicates.
    """

    # Start with accepted answers
    accepted_texts = {r["cooked"] for r in best_replies}
    final_replies = [r["cooked"] for r in best_replies]

    # Sort all scored replies by score (descending)
    sorted_replies = sorted(replies_data, key=lambda x: x[1], reverse=True)

    # Add top replies until we reach 5
    for reply_text, score in sorted_replies:
        if reply_text not in accepted_texts and len(final_replies) < 5:
            final_replies.append(reply_text)

    # If fewer than 5 replies exist in total, pad with what we have
    return final_replies[:5]

def vector_creation(data):
    encoder_question_data, replies_list = data
    score = encoder_question_data[1]
    post_id = encoder_question_data[2]
    vector_text = f"Question: {encoder_question_data[0][0]}\nTopic_Slug: {encoder_question_data[0][1]}"
    metadata = {
        f"Question": encoder_question_data[0][0],
        f"Topic_Slug": encoder_question_data[0][1],
        f"Score": score,
        f"Post_ID": post_id,
    }
    for i, reply in enumerate(replies_list):
        metadata[f"Reply {i+1}"] = reply

    # Use the global embedding model on CPU
    embedding_vector = EMBEDDING_MODEL.encode(
        vector_text,
        prompt_name="query",
        batch_size=1
    )

    return {
       "vector": embedding_vector.tolist(),
       "metadata": metadata
    }

def add_to_vector_databse(data_dict):
    client = QdrantClient(
        url="https://a9af870a-09da-4734-8536-26e6fbbed330.us-east4-0.gcp.cloud.qdrant.io:6333", 
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RfaG8hSt4RgVl9ehod1ejyLh_CVZF1Xu-C7OkqsRVRg",
    )
    id = str(uuid.uuid4())
    existing_collections = [c.name for c in client.get_collections().collections]
    if "chief-delphi-gpt" not in existing_collections:
        client.create_collection(
            collection_name="chief-delphi-gpt",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    new_point = PointStruct(id=id, vector=data_dict["vector"], payload=data_dict["metadata"])
    operation_info = client.upsert(
        collection_name="chief-delphi-gpt",
        wait=True,
        points=[new_point],
    )
    print(operation_info)


def main(args):
    filename = args.files[0]
    name = filename.split('.json')[0]
    inputFileName = name+'.json'
    if MAC or LINUX:
        with open(inputFileName, 'r') as inputFile:
            data = json.load(inputFile)
    else:
        with open(inputFileName, 'r', encoding='utf-8') as inputFile:
            data = json.load(inputFile)
    
    data = extractFeatures(data)
    data_dict = vector_creation(data)
    add_to_vector_databse(data_dict)
    print(data_dict)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON Parser")
    parser.add_argument('files', type=str, nargs='+', help='Input Clened JSON')
    args = parser.parse_args()
    main(args)
