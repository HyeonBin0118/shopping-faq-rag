"""
RAG 챗봇 3단계: LangChain RAG 파이프라인
=========================================
[변경 이력]
 v1 (Ollama 버전): llama3.2 로컬 LLM + HuggingFace 임베딩
 v2 (OpenAI 버전): GPT-4o-mini + text-embedding-3-small 

[변경 이유]
 - llama3.2 한국어 불안정: 태국어/필리핀어 혼용, 할루시네이션 발생
 - GPT-4o-mini: 한국어 안정적, 할루시네이션 감소, 응답 속도 향상
 - 임베딩/LLM 모두 OpenAI로 통일 → 생태계 일관성 확보

실행 전 설치:
    pip install langchain langchain-openai chromadb openai
"""

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── 설정 ──────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "shopping_rag"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
# ──────────────────────────────────────────────────


# ── 1. 임베딩 모델 (OpenAI) ────────────────────────
# v1에서 사용한 paraphrase-multilingual-MiniLM-L12-v2 대비
# text-embedding-3-small은 한국어/영어 혼합 검색 정확도가 더 높음
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
print("임베딩 모델: text-embedding-3-small (OpenAI)")


# ── 2. ChromaDB 연결 ───────────────────────────────
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# MMR유사도 + 다양성 균형 검색
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
print(f"ChromaDB 연결 완료: {vectorstore._collection.count()}개 벡터")


# ── 3. LLM 설정 (GPT-4o-mini) ─────────────────────
# temperature=0: 창의적 답변 억제 → 할루시네이션 최소화
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0
)
print("LLM: GPT-4o-mini (OpenAI)")


# ── 4. 프롬프트 템플릿 ─────────────────────────────
# [v1 → v2 변경점]
# v1: 한국어 지시문 → llama3.2가 불완전하게 따름 (타 언어 혼용)
# v2: 영어 지시문 유지 (GPT-4o-mini는 영어 지시도 한국어로 정확히 응답)
PROMPT_TEMPLATE = """You are a Korean shopping mall customer service chatbot.
Answer ONLY using the exact information from the reference documents below.
Rules:
- Answer in Korean only.
- NEVER invent product names, prices, or any information not in the documents.
- If documents contain relevant product info, quote it directly.
- If the answer is not in the documents, respond ONLY with: "해당 내용은 고객센터(1588-0000)로 문의해 주세요."
- Do NOT mix other languages into Korean sentences.

[Reference Documents]
{context}

[Customer Question]
{question}

[Answer]"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# ── 5. RAG 체인 구성 ───────────────────────────────
def format_docs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        cat = doc.metadata.get("category", "")
        parts.append(f"[문서{i} | {src} | {cat}]\n{doc.page_content}")
    return "\n\n".join(parts)


def get_source_info(docs: list) -> list:
    return [
        {
            "source": d.metadata.get("source"),
            "category": d.metadata.get("category"),
            "preview": d.page_content[:80] + "..."
        }
        for d in docs
    ]


# 한국어 상품 키워드 → 영어 번역 맵
# 상품 DB가 영어로 구성되어 있어 한국어 쿼리 검색 보완
KO_TO_EN = {
    "등산화": "hiking boots",
    "방수": "waterproof",
    "신발": "shoes",
    "자켓": "jacket",
    "부츠": "boots",
    "운동화": "sneakers",
    "샌들": "sandals",
}

def translate_query(question: str) -> str:
    """한국어 상품 키워드를 영어로 보완한 검색 쿼리 생성"""
    translated = question
    for ko, en in KO_TO_EN.items():
        if ko in translated:
            translated = translated.replace(ko, en)
    return translated


def ask(question: str, show_sources: bool = True) -> str:
    """
    RAG 파이프라인 메인 함수
    1. 질문 의도 파악 (상품 관련 여부)
    2. ChromaDB에서 관련 문서 검색
    3. source 필터링 (리뷰 오염 방지)
    4. GPT-4o-mini로 답변 생성
    """
    # 상품 관련 키워드 감지
    product_keywords = [
        "추천", "상품", "등산화", "신발", "자켓", "부츠", "옷", "의류",
        "boots", "shoes", "jacket", "hiking", "waterproof", "pants", "shirt"
    ]
    is_product_query = any(kw in question.lower() for kw in product_keywords)

    # 상품 질문은 한국어→영어 변환 후 검색 (상품 DB가 영어로 구성)
    # 일반 질문은 원문 그대로 검색
    search_query = translate_query(question) if is_product_query else question
    all_docs = vectorstore.similarity_search(search_query, k=20)

    if is_product_query:
        allowed = {"product", "faq"}   # 상품 질문 → 상품 + FAQ
    else:
        allowed = {"faq"}              # 일반 질문 → FAQ만

    docs = [d for d in all_docs if d.metadata.get("source") in allowed][:5]

    # 필터 후 결과 없으면 전체 사용 (fallback)
    if not docs:
        docs = all_docs[:5]

    chain = (
        {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)

    print(f"\n질문: {question}")
    print("─" * 50)
    print(f"답변: {answer}")

    if show_sources:
        print("\n📎 참고 문서:")
        for info in get_source_info(docs):
            print(f"  [{info['source']}|{info['category']}] {info['preview']}")

    return answer


# ── 6. 테스트 실행 ─────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "배송은 보통 며칠 걸려요?",
        "환불하고 싶은데 어떻게 해야 하나요?",
        "쿠폰이랑 적립금 같이 쓸 수 있나요?",
        "방수 등산화 추천해줘",
    ]

    print("=" * 60)
    print("RAG 챗봇 테스트 (v2 - OpenAI GPT-4o-mini)")
    print("=" * 60)

    for q in test_questions:
        ask(q)
        print()