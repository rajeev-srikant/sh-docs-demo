import os
import json
import glob
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- Settings ----------
EMBED_MODEL = "gemini-embedding-001"   # Gemini's embedding model
EMBED_DIM = 768                        # how many numbers per chunk
CHUNK_WORDS = 120                      # size of each chunk, in words
OVERLAP_WORDS = 25                     # words shared between neighbouring chunks


# ---------- Chunking: cut text into overlapping pieces ----------
def chunk_words(text, size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


# ---------- Steps 1 + 2: Load the PDFs and chunk them ----------
chunks = []
for path in sorted(glob.glob("docs/*.pdf")):
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in chunk_words(text, CHUNK_WORDS, OVERLAP_WORDS):
            chunks.append({
                "file": os.path.basename(path),
                "page": page_number,
                "text": piece,
            })

print(f"Created {len(chunks)} chunks from the PDFs")

# ---------- Step 3: Embed each chunk (turn text into numbers) ----------
texts = [c["text"] for c in chunks]
vectors = []
for i in range(0, len(texts), 20):          # send 20 chunks per request
    batch = texts[i:i + 20]
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=batch,
        config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": EMBED_DIM},
    )
    vectors.extend(e.values for e in result.embeddings)
    print(f"Embedded {len(vectors)} of {len(texts)} chunks")

for chunk, vector in zip(chunks, vectors):
    chunk["vector"] = vector

# ---------- Step 4: Save the index to a file ----------
with open("index.json", "w", encoding="utf-8") as f:
    json.dump({"embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM, "chunks": chunks}, f)

print("Saved index.json")
print("Example chunk:", chunks[0]["file"], "page", chunks[0]["page"])
print(chunks[0]["text"][:200], "...")