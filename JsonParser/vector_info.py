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

MAC = True
DEBUG = True
SCORE_CLIPPING = 1000
RECENCY_DECAY = 1080
MIN_TRUST_LEVEL = 0.1
Q_A_CLIPPINNG = 1500
question = None
vectors = []

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

if torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32

print(f"Loading model {MODEL_NAME} once on {DEVICE} ...")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True
).to(DEVICE)
MODEL.eval()
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
    global vectors
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
    replies = scoreReplies(replies)
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

    # Use the globally loaded TOKENIZER and MODEL
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
    parsed_replies_data = []
    for i, reply_data in enumerate(replies_data):
        if i == 0:
            parsed_replies_data.append(reply_data)
        if i < 4:
            if reply_data[1] < parsed_replies_data[0][1]:
                parsed_replies_data.append(reply_data)
            else:
                temp = [reply_data]
                parsed_replies_data = temp + parsed_replies_data
        if reply_data[1] < max(x[1] for x in parsed_replies_data):
            parsed_replies_data[0] = reply_data
    replies = [reply[0] for reply in parsed_replies_data]

    return replies

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
    
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    embedding_vector = model.encode(vector_text, prompt_name="query")
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
    if MAC:
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
