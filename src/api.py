import json
from pathlib import Path
from typing import List
import numpy as np
import faiss
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

BUILD_DIR = Path("build")
INDEX_PATH = BUILD_DIR / "index.faiss"
CHUNKS_PATH = BUILD_DIR / "chunks.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Mini RAG Search")

# Load at startup
index = faiss.read_index(str(INDEX_PATH))
chunks: List[dict] = [json.loads(x) for x in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
model = SentenceTransformer(MODEL_NAME)

class SearchHit(BaseModel):
    score: float
    doc_id: str
    chunk_idx: int
    text: str

class SearchResponse(BaseModel):
    query: str
    k: int
    hits: List[SearchHit]

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=2), k: int = Query(3, ge=1, le=20)):
    q_vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_vec, k)  # [1,k]
    scores, idxs = scores[0], idxs[0]

    hits: List[SearchHit] = []
    for s, i in zip(scores, idxs):
        if i < 0:  # FAISS returns -1 if fewer than k results exist
            continue
        c = chunks[int(i)]
        hits.append(SearchHit(score=float(s), doc_id=c["doc_id"], chunk_idx=c["chunk_idx"], text=c["text"]))
    return SearchResponse(query=q, k=k, hits=hits)
