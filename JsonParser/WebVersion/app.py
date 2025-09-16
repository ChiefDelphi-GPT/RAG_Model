import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

HOST_URL = "http://localhost:6333"
COLLECTION = "chief-delphi-gpt"
DEBUG = False

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
client = QdrantClient(url=HOST_URL)

def create_vector(user_question, user_topic):
    text = f"Question: {user_question}\nTopic_Slug: {user_topic}"
    return embedder.encode(text, prompt_name="query").tolist()

def extract_info(vector, top_k=50):
    return client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points

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

def build_prompt(top_items, user_question):
    prompt = ""
    for i, item in enumerate(top_items):
        prompt += f"Context {i+1}:\nQuestion: {item[0]}\nReplies:\n"
        for reply in item[3]:
            prompt += f"- {reply}\n"
        prompt += "\n"
    prompt += f"User's Question: {user_question}\nAnswer concisely:\n"
    return prompt

async def query_model(prompt: str, websocket: WebSocket):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32 if device == "mps" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True).to(device)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    def generate():
        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, generate)

    # Stream only model output, nothing else
    for token in streamer:
        await websocket.send_text(token)
    await websocket.send_text("[DONE]")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_question = data.get("question", "")
            user_topic = data.get("topic", "")

            vector = create_vector(user_question, user_topic)
            search_results = extract_info(vector, top_k=50)
            top_items = get_best_results(search_results)
            prompt = build_prompt(top_items, user_question)

            await query_model(prompt, websocket)

    except WebSocketDisconnect:
        if DEBUG:
            print("Client disconnected")
