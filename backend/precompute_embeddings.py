import json
import os

import pdfplumber
from sentence_transformers import SentenceTransformer

# Configuration
CORPUS_DIR = "Verified Indian Regulatory Document Corpus/A. RBI Master Directions-Foreeign Exchange Management"
OUTPUT_FILE = "../frontend/public/regulatory_embeddings.json"
CHUNK_SIZE = 500
OVERLAP = 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    # Clean text
    cleaned_text = " ".join(text.split()).strip()

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    start = 0
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk = cleaned_text[start:end].strip()
        if len(chunk) > 0:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def main():
    print("[1/5] Starting Regulatory Embeddings Pre-computation (Python Fallback)...")

    if not os.path.exists(CORPUS_DIR):
        print(f"ERROR: Corpus directory not found at {CORPUS_DIR}")
        return

    files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF documents.")

    print("[2/5] Loading SentenceTransformer Model (all-MiniLM-L6-v2)...")
    # This matches the Transformers.js model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_embeddings = []

    for i, file_name in enumerate(files):
        file_path = os.path.join(CORPUS_DIR, file_name)
        print(f"\n[3/5] Processing file {i + 1}/{len(files)}: {file_name}")

        try:
            # 1. Read PDF
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

            if not text.strip():
                print(f"Warning: No extractable text found in {file_name}")
                continue

            # 2. Chunk Text
            chunks = chunk_text(text)
            print(f"      Extracted text and split into {len(chunks)} chunks.")

            # 3. Generate Embeddings
            embeddings = model.encode(chunks, normalize_embeddings=True)

            for j, chunk in enumerate(chunks):
                # Ensure the vector is a list of floats
                vector = embeddings[j].tolist()

                all_embeddings.append({"text": chunk, "vector": vector, "documentName": file_name})

            print(f"      Successfully embedded {len(chunks)} chunks for {file_name}.")

        except Exception as e:
            print(f"      ERROR processing {file_name}: {e}")

    print(f"\n[4/5] Pre-computation complete. Total embeddings generated: {len(all_embeddings)}")

    print(f"[5/5] Saving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, ensure_ascii=False)

    print(f"\n✅ Success! Saved {len(all_embeddings)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
