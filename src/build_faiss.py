import json, argparse
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(jsonl_path: str):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out_dir", default="build")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.chunks)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(args.model)
    vecs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)       # cosine if normalized
    index.add(vecs)                    # order matches `chunks`

    faiss.write_index(index, str(out / "index.faiss"))
    np.save(out / "embeddings.npy", vecs)  # optional, handy for debugging
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Saved index + chunks to: {out.resolve()} | dim={d} | n={len(chunks)}")

if __name__ == "__main__":
    main()
