import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Iterable, Tuple, Optional

# --- Simple cleaners ---
def normalize_ws(text: str) -> str:
    # collapse whitespace + normalize newlines
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# --- Char-based chunker (fast & dependable) ---
def chunk_by_chars(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

# --- Optional: tokenizer-based chunker using tiktoken (closer to LLM limits) ---
def chunk_by_tokens(text: str, model: str = "cl100k_base", max_tokens: int = 400, overlap: int = 60) -> List[str]:
    try:
        import tiktoken
    except ImportError:
        raise SystemExit("Install tiktoken or use chunk_by_chars()")

    enc = tiktoken.get_encoding(model)
    toks = enc.encode(text)
    chunks = []
    start = 0
    n = len(toks)
    while start < n:
        end = min(n, start + max_tokens)
        piece = enc.decode(toks[start:end])
        chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return [normalize_ws(c) for c in chunks]

def load_text_files(folder: Path, exts=(".md", ".txt")) -> Dict[str, str]:
    out = {}
    for p in folder.glob("**/*"):
        if p.is_file() and p.suffix.lower() in exts:
            out[str(p)] = normalize_ws(p.read_text(encoding="utf-8", errors="ignore"))
    return out

def ingest(folder: str, mode: str = "chars", max_len: int = 800, overlap: int = 120) -> List[Dict]:
    folder_path = Path(folder)
    docs = load_text_files(folder_path)
    all_chunks = []
    for doc_id, text in docs.items():
        if mode == "tokens":
            chunks = chunk_by_tokens(text, max_tokens=max_len, overlap=overlap)
        else:
            chunks = chunk_by_chars(text, max_chars=max_len, overlap=overlap)
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_idx": i,
                "text": ch
            })
    return all_chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Folder with .md/.txt files")
    ap.add_argument("--mode", choices=["chars", "tokens"], default="chars")
    ap.add_argument("--max_len", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--out", default="chunks.jsonl")
    args = ap.parse_args()

    chunks = ingest(args.data, args.mode, args.max_len, args.overlap)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks → {args.out}")

if __name__ == "__main__":
    main()
