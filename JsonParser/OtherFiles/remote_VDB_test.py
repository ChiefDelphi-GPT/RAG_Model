from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


HOST_URL = "https://a9af870a-09da-4734-8536-26e6fbbed330.us-east4-0.gcp.cloud.qdrant.io"

def extract_info(vector_dict, top_k=50):
    client = QdrantClient(
    url="https://a9af870a-09da-4734-8536-26e6fbbed330.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RfaG8hSt4RgVl9ehod1ejyLh_CVZF1Xu-C7OkqsRVRg",
    )
    search_result = client.query_points(
        collection_name="chief-delphi-gpt",
        query=vector_dict["vector"],
        limit=top_k,
        with_payload=True,
    ).points

    return search_result

def query_vector_creation(user_question, user_topic):
    vector_text = f"Question: {user_question}\nTopic_Slug: {user_topic}"
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    embedding_vector = model.encode(vector_text, prompt_name="query")
    return {
        "vector": embedding_vector.tolist()
    }

if __name__ == "__main__":
    
    user_question = input("Enter your question: ")
    user_topic = input("Enter the topic slug: ")

    vector_dict = query_vector_creation(user_question, user_topic)
    search_results = extract_info(vector_dict, top_k=50)
    print(f"\nModel Response:\n{search_results}\n")