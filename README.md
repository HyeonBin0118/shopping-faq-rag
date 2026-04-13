# 쇼핑몰 RAG 챗봇

> FAQ, 상품 정보, 고객 리뷰를 기반으로 고객 질문에 자동 답변하는 RAG(Retrieval-Augmented Generation) 챗봇

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)

---

## 프로젝트 개요

### 문제 정의
쇼핑몰 고객센터는 배송, 환불, 상품 문의 등 반복적인 질문이 전체 문의의 70% 이상을 차지합니다. 이를 AI 챗봇으로 자동화하면 CS 운영 비용을 절감하고 24시간 즉각 응대가 가능해집니다.

### 해결 방법
단순 키워드 검색이 아닌 **의미 기반 검색(Semantic Search) + LLM 답변 생성**을 결합한 RAG 파이프라인을 구축했습니다.

### 주요 성과
- **1,784개** 벡터 DB 구축 (FAQ + 상품 + 리뷰 통합)
- **한국어/영어 혼합** 쿼리 검색 정확도 확보
- **멀티턴 대화** 지원 (이전 대화 문맥 기반 후속 질문 처리)
- **Ollama(무료) vs OpenAI API** 성능 비교 실험 수행 및 RAGAS 수치화

---

##  기술 스택

| 분류 | 기술 |
|---|---|
| 언어 | Python 3.11 |
| 임베딩 | `paraphrase-multilingual-MiniLM-L12-v2` / `text-embedding-3-small` |
| 벡터 DB | ChromaDB |
| LLM | Ollama llama3.2 / GPT-4o-mini |
| RAG 프레임워크 | LangChain |
| UI | Streamlit |
| 성능 평가 | RAGAS |

---

##  아키텍처

```
[데이터 소스]           [벡터 DB]              [RAG 파이프라인]
 FAQ (80개)    →                               
 상품 정보     →  청킹 → 임베딩 → ChromaDB  →  검색 → LLM → 답변
 고객 리뷰     →      (1,784개 벡터)            ↑
                                            대화 히스토리 (멀티턴)
```

---

## 프로젝트 구조

```
LLM/
├── chunks.jsonl                 # 전처리된 청크 데이터 (2,082개 → 중복 제거 후 1,784개)
├── faq.csv                      # FAQ 데이터 (80개)
├── productsclassified.csv       # 상품 데이터 (247개)
├── Reviews.csv                  # 고객 리뷰 데이터 (300개 샘플링)
├── chroma_db/                   # 벡터 DB (OpenAI 임베딩)
├── step1_data_prep.py           # 데이터 수집/전처리
├── step2_embedding.py           # OpenAI 임베딩 + ChromaDB 구축
├── step2_embedding_ollama.py    # Ollama용 HuggingFace 임베딩 (비교 실험용)
├── step3_rag_pipeline.py        # RAG 파이프라인 (CLI 테스트)
├── step4_streamlit_app.py       # 챗봇 UI (멀티턴 포함)
├── step6_ragas_compare.py       # RAGAS 성능 비교 평가
├── ragas_results.csv            # 평가 결과
└── requirements.txt
```

---

## 개발 과정

### 1단계 — 데이터 준비

**수집 데이터**
- FAQ 80개: 배송/결제/주문/환불/회원/쿠폰/상품/고객센터 8개 카테고리 직접 작성
- 상품 데이터: Kaggle `productsclassified.csv` (247개 상품)
- 고객 리뷰: Kaggle Amazon Reviews (568,454개 중 유용도 상위 300개 샘플링)

**전처리**
```python
# 텍스트 정제
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)      # HTML 태그 제거
    text = re.sub(r'\s+', ' ', text).strip() # 공백 정리
    return text

# 청킹 (chunk_size=500, overlap=50)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
```

**결과물: `chunks.jsonl` — 총 2,082개 청크 (중복 제거 후 1,784개)**

| source | 청크 수 |
|---|---|
| FAQ | 80개 |
| 상품 | 643개 |
| 리뷰 | 1,359개 → 중복 제거 후 1,061개 |

---

### 2단계 — 임베딩 + 벡터 DB 구축

**임베딩 모델 선택 및 전환**

초기에는 `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace 무료)를 사용했으나, LLM을 GPT-4o-mini로 교체하면서 **임베딩도 OpenAI `text-embedding-3-small`로 통일**했습니다.

전환 이유:
- LLM과 임베딩을 같은 OpenAI 생태계로 통일하여 벡터 공간의 일관성 확보
- 1,784개 기준 재임베딩 비용 약 **$0.003** (사실상 무료)

**임베딩 모델 유사도 점수 비교**

| 쿼리 | MiniLM (v1) | text-embedding-3-small (v2) |
|---|---|---|
| "배송 며칠 걸려요?" | 0.68 | 0.62 |
| "환불 정책이 어떻게 돼요?" | 0.76 | 0.43 |
| "hiking boots waterproof" | 0.75 | 0.64 |

> **인사이트:** MiniLM은 다국어 특화 모델이라 한국어 쿼리에서 유사도 점수가 더 높게 나왔다. 그러나 단순 유사도 점수보다 실제 답변 품질을 반영하는 **RAGAS 수치(Context Precision +12.5%, Context Recall +12.5%)** 에서 text-embedding-3-small이 전체적으로 우세했다. 또한 LLM과 같은 OpenAI 생태계로 통일하여 벡터 공간의 일관성을 확보하는 것이 전환의 주된 이유였다.

**ChromaDB 설정**
```python
collection = client.create_collection(
    name="shopping_rag",
    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
)
```

---

### 3단계 — RAG 파이프라인 구축

**LangChain RAG 체인 구성**
```
사용자 질문 → 임베딩 → ChromaDB 검색(MMR, k=5) → 프롬프트 조합 → LLM → 답변
```

**소스 필터링 전략**

리뷰 데이터(영어)가 한국어 FAQ 검색을 오염시키는 문제를 발견하여 질문 의도에 따라 검색 소스를 동적으로 필터링하는 방식을 적용했습니다.

```python
# 상품 관련 질문 → faq + product 검색
# 일반 질문 → faq만 검색
if is_product_query:
    allowed = {"product", "faq"}
else:
    allowed = {"faq"}
```

**한국어↔영어 언어 불일치 해결**

상품 DB가 영어로 구성되어 있어 "방수 등산화 추천해줘" 같은 한국어 쿼리로는 관련 문서를 찾지 못하는 문제가 있었습니다. **키워드 번역 맵**을 구현하여 상품 관련 질문 감지 시 자동으로 영어 쿼리로 변환하여 검색 정확도를 향상시켰습니다.

```python
KO_TO_EN = {
    "등산화": "hiking boots", "방수": "waterproof",
    "신발": "shoes", "자켓": "jacket", ...
}
```

**프롬프트 엔지니어링**

초기에는 한국어로 지시문을 작성했으나, llama3.2가 한국어 지시를 불완전하게 따르는 현상 발견(태국어/필리핀어 혼용). **영어 지시문으로 변경 후 할루시네이션이 감소**했습니다.

```python
PROMPT_TEMPLATE = """You are a Korean shopping mall customer service chatbot.
Answer ONLY using the exact information from the reference documents below.
Rules:
- Answer in Korean only.
- NEVER invent product names, prices, or any information not in the documents.
- If the answer is not in the documents, respond ONLY with: "해당 내용은 고객센터(1588-0000)로 문의해 주세요."
..."""
```

---

### 4단계 — Streamlit UI

**주요 기능**
- 사이드바: 프로젝트 소개, 기술 스택 뱃지, 데이터 현황, 빠른 질문 버튼 6개
- 메인: 다크 헤더 + 버블 형태 채팅 UI + 참고 문서 출처 태그
- `@st.cache_resource`로 모델 초기화 캐싱 → 재질문 시 응답 속도 향상

---

### 5단계 — 멀티턴 대화 지원

단순 1회성 질문/답변에서 나아가 **이전 대화 문맥을 기억하는 멀티턴 대화**를 구현했습니다.

**핵심 로직 2가지**

1. **대화 히스토리 프롬프트 포함** — 최근 3턴의 대화를 프롬프트에 포함시켜 후속 질문 처리
2. **후속 질문 감지** — "그거", "그중", "방금" 등 지시어 감지 시 이전 대화의 키워드를 검색 쿼리에 합쳐 ChromaDB 검색 정확도 향상

**멀티턴 테스트 결과**
```
질문1: "방수 등산화 추천해줘"
→ Men's Trail Model 4, Men's Vasque Talus Trek 등 5개 추천

질문2: "그 중에 제일 가벼운건 뭐야?"
→ Women's Merrell Moab 2 Waterproof Hiking Boots (1 lb. 3 oz.) 정확히 답변
```

---

### 6단계 — RAGAS 성능 비교 평가

동일한 8개 테스트 질문으로 두 버전을 정량적으로 비교했습니다.

**평가 지표**
| 지표 | 설명 |
|---|---|
| Faithfulness | 답변이 검색 문서에 충실한가 (할루시네이션 측정) |
| Answer Relevancy | 질문과 답변이 관련 있는가 |
| Context Precision | 검색된 문서가 질문에 적합한가 |
| Context Recall | 필요한 문서를 빠짐없이 가져왔는가 |

**비교 결과**

| 지표 | Ollama v1 (llama3.2 + MiniLM) | OpenAI v2 (GPT-4o-mini + text-embedding-3-small) | 개선 |
|---|---|---|---|
| Faithfulness | 0.8125 | **0.8750** | +6.3% |
| Answer Relevancy | 0.4370 | **0.4667** | +3.0% |
| Context Precision | 0.7500 | **0.8750** | +12.5% |
| Context Recall | 0.7500 | **0.8750** | +12.5% |

> **결론:** OpenAI 버전이 모든 지표에서 우세하며, 특히 Context Precision/Recall이 +12.5% 향상되었습니다. 검색 단계의 품질이 전체 RAG 성능에 가장 큰 영향을 미친다는 점을 수치로 확인했습니다.

---

## 실험: Ollama(llama3.2) vs OpenAI(GPT-4o-mini) 비교

| 항목 | llama3.2 (Ollama) | GPT-4o-mini (OpenAI) |
|---|---|---|
| 비용 | 무료 | ~$0.001/회 |
| 한국어 품질 | 간헐적 타 언어 혼용 | 안정적 |
| 할루시네이션 | 발생 (없는 상품 생성) | 거의 없음 |
| 응답 속도 | 로컬 GPU 의존 | 빠름 |
| 인터넷 필요 | 불필요 | 필요 |
| RAGAS Faithfulness | 0.8125 | **0.8750** |
| RAGAS Context Precision | 0.7500 | **0.8750** |

> **결론:** 프로덕션 환경에서는 GPT-4o-mini, 비용 제약 환경에서는 llama3.2 권장

---

##  설치 및 실행

```bash
# 1. 환경 설정
conda create -n rag_env python=3.11
conda activate rag_env
pip install -r requirements.txt

# 2. API 키 설정
set OPENAI_API_KEY=sk-...   # Windows
export OPENAI_API_KEY=sk-... # Mac/Linux

# 3. 데이터 준비 (청크 생성)
python step1_data_prep.py

# 4. 벡터 DB 구축 (OpenAI 임베딩)
python step2_embedding.py

# 5. RAG 파이프라인 CLI 테스트
python step3_rag_pipeline.py

# 6. Streamlit UI 실행
streamlit run step4_streamlit_app.py

# 7. RAGAS 성능 비교 (선택)
python step6_ragas_compare.py
```

---

## 배운 점 및 개선 방향

**배운 점**
- 데이터 품질이 검색 정확도에 직결됨 (리뷰 데이터 오염 문제 직접 경험)
- 프롬프트 언어(한국어 vs 영어)에 따라 LLM 응답 품질이 달라짐
- 임베딩 모델 선택 시 데이터의 언어 분포를 고려해야 함
- RAGAS로 성능을 수치화하면 모델 교체의 근거를 데이터로 제시할 수 있음

**개선 방향**
- 리뷰 데이터 한국어 번역 후 재임베딩
- 청크 사이즈 실험 (300/500/700 비교)
- Re-ranking 기법 적용으로 검색 정확도 향상
- Streamlit Cloud 배포

---

## API 사용 비용

| 항목 | 비용 |
|---|---|
| 임베딩 1,784개 (text-embedding-3-small) | ~$0.003 |
| RAG 테스트 쿼리 (gpt-4o-mini) | ~$0.01 |
| RAGAS 평가 (gpt-4o-mini) | ~$0.05 |
| **합계** | **~$0.063 (약 90원)** |

---

## 참고 자료

- [LangChain 공식 문서](https://docs.langchain.com)
- [ChromaDB 공식 문서](https://docs.trychroma.com)
- [RAGAS 공식 문서](https://docs.ragas.io)
- [Sentence Transformers](https://www.sbert.net)
- 데이터셋: [Kaggle Amazon Reviews](https://www.kaggle.com), [Kaggle E-commerce Products](https://www.kaggle.com)
