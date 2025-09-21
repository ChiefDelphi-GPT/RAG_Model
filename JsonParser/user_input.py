import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess, sys, os

 
HOST_URL = "http://localhost:6333"
DEBUG = False

def install_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

def query_vector_creation(user_question, user_topic):
    vector_text = f"Question: {user_question}\nTopic_Slug: {user_topic}"
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    embedding_vector = model.encode(vector_text, prompt_name="query")
    return {
        "vector": embedding_vector.tolist()
    }

def extract_info(vector_dict, top_k=50):
    client = QdrantClient(url=HOST_URL)
    search_result = client.query_points(
        collection_name="chief-delphi-gpt",
        query=vector_dict["vector"],
        limit=top_k,
        with_payload=True,
    ).points

    if DEBUG:
        print(f"Found {len(search_result)} vectors")
        for point in search_result:
            print(f"ID: {point.id}, Score: {point.score}, Payload: {point.payload}")

    return search_result

def get_best_results(search_results):
    return sorted(
        (
            (
                p.payload["Question"],
                p.score * p.payload.get("Score", 1),
                p.payload["Post_ID"],
                [v.strip() for k, v in p.payload.items() if k.startswith("Reply_")]
            )
            for p in search_results
        ),
        key=lambda x: x[1],
        reverse=True
    )[:5]

def queryDeepSeek(input_text):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # Smallest - fastest loading
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"    # Good balance of quality and resource usage
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # Alternative 8B option
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"   # Larger for better reasoning
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"   # High-end consumer hardware
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # Very large - needs lots of RAM
    # model_name = "deepseek-ai/DeepSeek-R1"                     # Main flagship - enterprise only (671B)

    if (torch.backends.mps.is_available()):
        device = "mps"
        torch_dtype = torch.float32  # MPS works better with float32
    elif (torch.cuda.is_available()):
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,  # Don't use auto device mapping
        trust_remote_code=True  # DeepSeek models may require this
    ).to(device)

    messages = [
        {"role": "user", "content": input_text}
    ]
    try: 
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        formatted_input = f"User: {input_text}\nAssistant:"

    start_time = time.time()
    input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10000,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )

    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    end_time = time.time()
    return response.strip(), end_time - start_time

def feed_through_model(top_items, user_question):
    prompt = "You are an expert assistant for the Chief Delphi forums, helping users find accurate and relevant information based on their questions. Use the provided context from previous discussions to formulate a comprehensive and helpful response to the user's question.\n\n"
    for i, item in enumerate(top_items):
        prompt += f"Context {i+1}:\nQuestion: {item[0]}\nReplies:\n"
        for reply in item[3]:
            prompt += f"- {reply}\n"
        prompt += "\n"
    prompt += f"User's Question: {user_question}\n\n"
    prompt += "Based on the above contexts, provide a detailed and accurate answer to the user's question. If the information is insufficient, indicate that more details are needed.\n"
    response, duration = queryDeepSeek(prompt)
    if DEBUG:
        print(f"Model response time: {duration:.2f} seconds")
    return response

if __name__ == "__main__":
    # THIS IS THE LINE THAT AUTO INSTALLS THE REQUIREMENTS
    #--------------------------------
    install_requirements()
    #--------------------------------
    user_question = input("Enter your question: ")
    user_topic = input("Enter the topic slug: ")

    vector_dict = query_vector_creation(user_question, user_topic)
    search_results = extract_info(vector_dict, top_k=50)
    top_items = get_best_results(search_results)
    final_response = feed_through_model(top_items, user_question)
    print(f"\nModel Response:\n{final_response}\n")

