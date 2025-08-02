import fitz  # PyMuPDF
import os
import tiktoken

def extract_text_from_pdfs(pdf_folder="data/pdfs"):
    text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(pdf_folder, filename)) as doc:
                for page in doc:
                    text += page.get_text()
    return text

def chunk_text(text, max_tokens=250):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

if __name__ == "__main__":
    text = extract_text_from_pdfs()
    chunks = chunk_text(text)
    print(f"Extracted {len(chunks)} chunks.")
