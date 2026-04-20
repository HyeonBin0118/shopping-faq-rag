"""
RAG 챗봇 6단계: RAGAS 성능 비교
=================================
Ollama(llama3.2 + MiniLM) vs OpenAI(GPT-4o-mini + text-embedding-3-small)
두 버전의 RAG 파이프라인을 동일한 테스트셋으로 평가

평가 지표:
  - Faithfulness: 답변이 검색 문서에 충실한가 (할루시네이션 측정)
  - Answer Relevancy: 질문과 답변이 관련 있는가
  - Context Precision: 검색된 문서가 질문에 적합한가
  - Context Recall: 필요한 문서를 빠짐없이 가져왔는가

실행 전 설치:
    pip install ragas langchain-chroma sentence-transformers
"""

import os
import json
import pandas as pd
from datasets import Dataset

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── 설정 ──────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
# ──────────────────────────────────────────────────


# ── 테스트셋 ───────────────────────────────────────
# 정답(ground_truth)이 있어야 RAGAS가 평가 가능
# FAQ 기반으로 정답을 직접 작성
TEST_SET = [
    {
        "question": "배송은 보통 며칠 걸려요?",
        "ground_truth": "일반적으로 결제 완료 후 2~3 영업일 내에 배송됩니다."
    },
    {
        "question": "환불하고 싶은데 어떻게 해야 하나요?",
        "ground_truth": "마이페이지 > 반품/교환 신청 후 상품을 반송하시면 확인 후 환불됩니다."
    },
    {
        "question": "쿠폰이랑 적립금 같이 쓸 수 있나요?",
        "ground_truth": "쿠폰과 적립금은 동시에 사용하실 수 있습니다."
    },
    {
        "question": "주말에도 배송되나요?",
        "ground_truth": "주말 및 공휴일에는 배송이 이루어지지 않습니다."
    },
    {
        "question": "환불 기간은 얼마나 걸리나요?",
        "ground_truth": "상품 회수 후 2~5 영업일 내에 환불 처리됩니다."
    },
    {
        "question": "배송비는 얼마예요?",
        "ground_truth": "기본 배송비는 3,000원이며, 일부 상품은 무료배송이 적용됩니다."
    },
    {
        "question": "적립금은 어떻게 사용하나요?",
        "ground_truth": "결제 페이지에서 사용할 적립금 금액을 입력하시면 됩니다."
    },
    {
        "question": "회원가입 없이 주문할 수 있나요?",
        "ground_truth": "비회원 주문도 가능합니다. 주문 시 이메일을 입력하시면 주문 내역을 확인할 수 있습니다."
    },
]


# ── 프롬프트 ───────────────────────────────────────
PROMPT_TEMPLATE = """You are a Korean shopping mall customer service chatbot.
Answer ONLY using the exact information from the reference documents below.
Rules:
- Answer in Korean only.
- NEVER invent information not in the documents.
- If the answer is not in the documents, respond ONLY with: "해당 내용은 고객센터(1588-0000)로 문의해 주세요."

[Reference Documents]
{context}

[Customer Question]
{question}

[Answer]"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# ── RAG 파이프라인 실행 함수 ───────────────────────
def run_rag(question, vectorstore, llm):
    """단일 질문에 대해 RAG 실행 → 답변 + 검색 문서 반환"""
    docs = vectorstore.similarity_search(question, k=5)
    # FAQ만 검색
    docs = [d for d in docs if d.metadata.get("source") == "faq"][:5]
    if not docs:
        docs = vectorstore.similarity_search(question, k=5)

    context = "\n\n".join([
        f"[문서{i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    ])

    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    contexts = [d.page_content for d in docs]
    return answer, contexts


def build_ragas_dataset(vectorstore, llm, version_name):
    """테스트셋 전체 실행해서 RAGAS 입력 데이터셋 구성"""
    print(f"\n{'='*50}")
    print(f"{version_name} 테스트 실행 중...")
    print(f"{'='*50}")

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
        q = item["question"]
        print(f"  질문: {q}")
        answer, ctx = run_rag(q, vectorstore, llm)
        print(f"  답변: {answer[:60]}...")

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


# ── 메인 실행 ──────────────────────────────────────
if __name__ == "__main__":

    # RAGAS 평가에 사용할 LLM (GPT-4o-mini 기준으로 통일)
    # 평가 자체는 동일한 LLM으로 해야 공정함
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    openai_embeddings_eval = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    results = {}

    # ── v1: Ollama 버전 ────────────────────────────
    print("\n[1/2] Ollama 버전 (llama3.2 + MiniLM) 초기화...")
    try:
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        vs_ollama = Chroma(
            collection_name="shopping_rag_ollama",
            embedding_function=hf_embeddings,
            persist_directory=CHROMA_DIR,
        )
        ollama_llm = OllamaLLM(model="llama3.2")
        ollama_llm.invoke("hi")  # 연결 확인
        print(f"  벡터 수: {vs_ollama._collection.count()}개")

        dataset_ollama = build_ragas_dataset(vs_ollama, ollama_llm, "Ollama v1")
        result_ollama = evaluate(
            dataset_ollama,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=openai_embeddings_eval,
        )
        results["Ollama (llama3.2 + MiniLM)"] = result_ollama

    except Exception as e:
        print(f"  ⚠ Ollama 오류: {e}")

    # ── v2: OpenAI 버전 ────────────────────────────
    print("\n[2/2] OpenAI 버전 (GPT-4o-mini + text-embedding-3-small) 초기화...")
    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    vs_openai = Chroma(
        collection_name="shopping_rag",
        embedding_function=openai_embeddings,
        persist_directory=CHROMA_DIR,
    )
    openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    print(f"  벡터 수: {vs_openai._collection.count()}개")

    dataset_openai = build_ragas_dataset(vs_openai, openai_llm, "OpenAI v2")
    result_openai = evaluate(
        dataset_openai,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=openai_embeddings_eval,
    )
    results["OpenAI (GPT-4o-mini + text-embedding-3-small)"] = result_openai

    # ── 결과 출력 ──────────────────────────────────
    print("\n" + "="*60)
    print("RAGAS 성능 비교 결과")
    print("="*60)

    def get_score(result, key):
        """RAGAS 버전 상관없이 점수 추출"""
        val = result[key]
        if isinstance(val, list):
            # 최신 RAGAS: 리스트로 반환 → 평균 계산
            val = [v for v in val if v is not None]
            return round(sum(val) / len(val), 4) if val else 0.0
        return round(float(val), 4)

    rows = []
    for version, result in results.items():
        row = {
            "버전": version,
            "Faithfulness": get_score(result, "faithfulness"),
            "Answer Relevancy": get_score(result, "answer_relevancy"),
            "Context Precision": get_score(result, "context_precision"),
            "Context Recall": get_score(result, "context_recall"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # CSV 저장
    df.to_csv("ragas_results.csv", index=False, encoding="utf-8-sig")
    print("\n 결과 저장 완료: ragas_results.csv")
    
