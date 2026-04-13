"""
RAG 챗봇 2단계 (Ollama 버전): HuggingFace 임베딩 + ChromaDB
=============================================================
RAGAS 비교 실험용 — OpenAI 버전과 별도 컬렉션으로 저장
    shopping_rag_ollama  ← 이 파일로 생성
    shopping_rag         ← OpenAI 버전 (기존)

실행 전 설치:
    pip install sentence-transformers chromadb tqdm
"""

import json
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── 설정 ──────────────────────────────────────────
CHUNKS_PATH = "chunks.jsonl"
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "shopping_rag_ollama"   # OpenAI 버전과 구분
BATCH_SIZE  = 64
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# ──────────────────────────────────────────────────

model = SentenceTransformer(EMBED_MODEL)

def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=False).tolist()

def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    seen = set()
    unique = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            unique.append(c)
    print(f"청크 로드: {len(chunks)}개 → 중복 제거 후 {len(unique)}개")
    return unique

def build_vectordb(chunks):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"기존 컬렉션 '{COLLECTION}' 삭제")

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="HuggingFace 임베딩 저장 중"):
        batch     = chunks[i : i + BATCH_SIZE]
        texts     = [c["text"]   for c in batch]
        ids       = [c["doc_id"] for c in batch]
        metadatas = [{"source": c["source"], "category": c["category"]} for c in batch]
        embeddings = get_embeddings(texts)
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    print(f"\n✅ 저장 완료! {collection.count()}개 벡터 → 컬렉션: {COLLECTION}")
    return collection

if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    build_vectordb(chunks)
    print("\nOllama용 DB 구축 완료! 이제 step6_ragas_compare.py 실행하세요.")
