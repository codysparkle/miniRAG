import json, argparse, textwrap, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Hugging Face sentence embeddings
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

def load_chunks(jsonl_path: str) -> List[Dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # safe cosine for non-zero vectors
    denom = (norm(a) * norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def build_or_load_embeddings(chunks: List[Dict], model_name: str, cache_dir: str = ".cache") -> np.ndarray:
    Path(cache_dir).mkdir(exist_ok=True)
    cache_path = Path(cache_dir) / f"embeddings_{Path(model_name).name}.npy"
    text_sig_path = Path(cache_dir) / "embeddings_text_count.txt"

    texts = [c["text"] for c in chunks]

    # reuse cache if text count matches
    if cache_path.exists() and text_sig_path.exists():
        with open(text_sig_path, "r") as f:
            if f.read().strip() == str(len(texts)):
                return np.load(cache_path)

    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    np.save(cache_path, vecs)
    with open(text_sig_path, "w") as f:
        f.write(str(len(texts)))
    return vecs

def search(query: str, chunk_vecs: np.ndarray, chunks: List[Dict], model_name: str, k: int = 3) -> List[Tuple[float, int]]:
    model = SentenceTransformer(model_name)
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = chunk_vecs @ q_vec  # cosine if vectors are normalized
    idxs = np.argsort(-sims)[:k]
    return [(float(sims[i]), int(i)) for i in idxs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("query", nargs="+", help="Your search query")
    args = ap.parse_args()

    query = " ".join(args.query)
    chunks = load_chunks(args.chunks)
    vecs = build_or_load_embeddings(chunks, args.model)

    results = search(query, vecs, chunks, args.model, k=args.k)

    print(f"\nQuery: {query}\nTop {args.k} results (dense embeddings):")
    for rank, (score, idx) in enumerate(results, 1):
        c = chunks[idx]
        preview = textwrap.shorten(c["text"], width=140, placeholder=" ...")
        print(f"\n[{rank}] score={score:.3f}  source={c['doc_id']}  chunk#{c['chunk_idx']}")
        print(preview)

    context = "\n\n".join(chunks[i]["text"] for _, i in results)
    prompt = f"""You are a helpful assistant. Use ONLY the provided context.
Question: {query}

Context:
{context}

Answer:"""
    print("\n--- LLM-ready prompt (preview) ---")
    print(textwrap.shorten(prompt, width=500, placeholder=" ..."))

if __name__ == "__main__":
    main()
