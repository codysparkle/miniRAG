# Mini RAG (1–2 hrs/day, 5 days)

A tiny Retrieval-Augmented Generation skeleton:
- Ingest + chunk local `.md/.txt`
- TF-IDF baseline & sentence-embedding retrieval
- FAISS index
- Minimal FastAPI `/search` endpoint

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/ingest.py --data data --mode chars --max_len 600 --overlap 100
python src/build_faiss.py --chunks chunks.jsonl
uvicorn src.api:app --reload --port 8000
