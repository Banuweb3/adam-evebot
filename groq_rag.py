# import openai
# import faiss
# import numpy as np
# import pickle
# from dotenv import load_dotenv
# import os

# load_dotenv()
# openai.api_key = os.getenv("GROQ_API_KEY")
# EMBED_MODEL = "text-embedding-ada-002"
# LLM_MODEL = "llama3-70b-8192"

# def get_embedding(text):
#     response = openai.Embedding.create(input=[text], model=EMBED_MODEL)
#     return response['data'][0]['embedding']

# def load_faiss_index(index_path="data/faiss_index"):
#     index = faiss.read_index(os.path.join(index_path, "index.faiss"))
#     with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
#         chunks = pickle.load(f)
#     return index, chunks

# def search_similar_chunks(query, top_k=3):
#     query_vec = np.array(get_embedding(query)).astype('float32').reshape(1, -1)
#     index, chunks = load_faiss_index()
#     distances, indices = index.search(query_vec, top_k)
#     return [chunks[i] for i in indices[0]]

# def generate_answer(query):
#     context_chunks = search_similar_chunks(query)
#     context = "\n\n".join(context_chunks)
#     prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

#     response = openai.ChatCompletion.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response['choices'][0]['message']['content']



# import os
# import pickle
# import numpy as np
# import faiss
# from dotenv import load_dotenv
# from groq import Groq  # ✅ Groq for LLM
# from sentence_transformers import SentenceTransformer  # ✅ For embeddings

# # Load environment variables
# load_dotenv()
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # ✅ Groq API

# # Models
# LLM_MODEL = "llama3-70b-8192"  # ✅ Groq-supported LLM

# # Load sentence-transformer model once
# embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ✅ Free and accurate

# # Get embedding using local model
# def get_embedding(text):
#     return embedder.encode(text).tolist()

# # Load FAISS index and chunks
# def load_faiss_index(index_path="data/faiss_index"):
#     index = faiss.read_index(os.path.join(index_path, "index.faiss"))
#     with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
#         chunks = pickle.load(f)
#     return index, chunks

# # Search top-k relevant chunks
# def search_similar_chunks(query, top_k=3):
#     query_vec = np.array(get_embedding(query)).astype('float32').reshape(1, -1)
#     index, chunks = load_faiss_index()
#     distances, indices = index.search(query_vec, top_k)
#     return [chunks[i] for i in indices[0]]

# # Generate answer using Groq's LLM
# def generate_answer(query):
#     context_chunks = search_similar_chunks(query)
#     context = "\n\n".join(context_chunks)
#     prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

#     response = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content.strip()







import os
import pickle
import numpy as np
import faiss
import re  # ✅ For markdown cleanup
from dotenv import load_dotenv
from groq import Groq  # ✅ Groq for LLM
from sentence_transformers import SentenceTransformer  # ✅ For embeddings

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # ✅ Groq API

# Models
LLM_MODEL = "llama3-70b-8192"  # ✅ Groq-supported LLM

# Load sentence-transformer model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ✅ Free and accurate

# Get embedding using local model
def get_embedding(text):
    return embedder.encode(text).tolist()

# Load FAISS index and chunks
def load_faiss_index(index_path="data/faiss_index"):
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Search top-k relevant chunks
def search_similar_chunks(query, top_k=3):
    query_vec = np.array(get_embedding(query)).astype('float32').reshape(1, -1)
    index, chunks = load_faiss_index()
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# ✅ Function to clean markdown
def clean_markdown(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # Remove italic
    text = re.sub(r"__(.*?)__", r"\1", text)      # Remove __bold__
    text = re.sub(r"_([^_]+)_", r"\1", text)      # Remove _italic_
    return text

# Generate answer using Groq's LLM
def generate_answer(query):
    context_chunks = search_similar_chunks(query)
    context = "\n\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    raw_answer = response.choices[0].message.content.strip()
    return clean_markdown(raw_answer)
