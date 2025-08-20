import json, argparse, textwrap
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_chunks(jsonl_path: str) -> List[Dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def build_index(texts: List[str]) -> Tuple[TfidfVectorizer, any]:
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(texts)  # [n_chunks x n_features]
    return vec, X

def search(query: str, vec: TfidfVectorizer, X, k: int = 3) -> List[Tuple[float, int]]:
    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    idxs = sims.argsort()[::-1][:k]
    return [(float(sims[i]), int(i)) for i in idxs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("query", nargs="+", help="Your search query")
    args = ap.parse_args()

    query = " ".join(args.query)
    chunks = load_chunks(args.chunks)
    texts = [c["text"] for c in chunks]

    vec, X = build_index(texts)
    results = search(query, vec, X, k=args.k)

    print(f"\nQuery: {query}\nTop {args.k} results:")
    for rank, (score, idx) in enumerate(results, 1):
        c = chunks[idx]
        preview = textwrap.shorten(c["text"], width=140, placeholder=" ...")
        print(f"\n[{rank}] score={score:.3f}  source={c['doc_id']}  chunk#{c['chunk_idx']}")
        print(preview)

    # Optional: build an LLM-ready prompt
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
