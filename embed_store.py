# import openai
# import faiss
# import os
# import pickle
# from dotenv import load_dotenv
# from pdf_loader import extract_text_from_pdfs, chunk_text

# load_dotenv()
# openai.api_key = os.getenv("GROQ_API_KEY")
# EMBED_MODEL = "text-embedding-ada-002"

# def get_embedding(text):
#     response = openai.Embedding.create(input=[text], model=EMBED_MODEL)
#     return response['data'][0]['embedding']

# def build_faiss_index(chunks, index_path="data/faiss_index"):
#     embeddings = [get_embedding(chunk) for chunk in chunks]
#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings).astype('float32'))

#     os.makedirs(index_path, exist_ok=True)
#     faiss.write_index(index, os.path.join(index_path, "index.faiss"))

#     with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
#         pickle.dump(chunks, f)

# if __name__ == "__main__":
#     raw_text = extract_text_from_pdfs()
#     chunks = chunk_text(raw_text)
#     build_faiss_index(chunks)
#     print(f"FAISS index created with {len(chunks)} chunks.")



import faiss
import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer  # ✅ Free local embeddings
from pdf_loader import extract_text_from_pdfs, chunk_text

# Load environment variables (if needed)
load_dotenv()

# Load sentence-transformer model (free)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get embeddings using sentence-transformers
def get_embedding(text):
    return model.encode(text)

# Build FAISS index from chunks
def build_faiss_index(chunks, index_path="data/faiss_index"):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))

    with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    raw_text = extract_text_from_pdfs(pdf_folder="data/pdfs")
    chunks = chunk_text(raw_text)
    build_faiss_index(chunks)
    print(f"✅ FAISS index created with {len(chunks)} chunks.")

