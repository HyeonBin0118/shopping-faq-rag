# 쇼핑몰 RAG 챗봇 포트폴리오

> 쇼핑몰 FAQ, 상품 정보, 고객 리뷰를 기반으로 고객 질문에 자동 답변하는 RAG(Retrieval-Augmented Generation) 챗봇

---

## 프로젝트 개요

### 문제 정의
쇼핑몰 고객센터는 배송, 환불, 상품 문의 등 반복적인 질문이 전체 문의의 70% 이상을 차지합니다.
이를 AI 챗봇으로 자동화하면 CS 운영 비용을 절감하고 24시간 즉각 응대가 가능해집니다.

### 해결 방법
단순 키워드 검색이 아닌 **의미 기반 검색(Semantic Search)** + **LLM 답변 생성**을 결합한 RAG 파이프라인을 구축했습니다.

### 주요 성과
- 1,784개 벡터 DB 구축 (FAQ + 상품 + 리뷰 통합)
- 한국어/영어 혼합 쿼리 검색 정확도 확보
- Ollama(무료) vs OpenAI API 성능 비교 실험 수행

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.11 |
| 임베딩 | `paraphrase-multilingual-MiniLM-L12-v2` / `text-embedding-3-small` |
| 벡터 DB | ChromaDB |
| LLM | Ollama llama3.2 / GPT-4o-mini |
| RAG 프레임워크 | LangChain |
| UI | Streamlit (4단계) |

---

## 아키텍처

```
[데이터 소스]          [벡터 DB]           [RAG 파이프라인]
 FAQ (80개)    →                          
 상품 정보     →  청킹 → 임베딩 → ChromaDB → 검색 → LLM → 답변
 고객 리뷰     →      (1,784개 벡터)
```

---

## 개발 과정

### 1단계 — 데이터 준비 (1주)

**수집 데이터**
- FAQ 80개: 배송/결제/주문/환불/회원/쿠폰/상품/고객센터 8개 카테고리 직접 작성
- 상품 데이터: Kaggle `productsclassified.csv` (247개 상품)
- 고객 리뷰: Kaggle `Amazon Reviews` (568,454개 중 유용도 상위 300개 샘플링)

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

**결과물:** `chunks.jsonl` — 총 2,082개 청크 (중복 제거 후 1,784개)

| source | 청크 수 |
|--------|--------|
| FAQ | 80개 |
| 상품 | 643개 |
| 리뷰 | 1,359개 |

---

### 2단계 — 임베딩 + 벡터 DB 구축 (1주)

**임베딩 모델 선택 이유**

`paraphrase-multilingual-MiniLM-L12-v2`를 선택한 이유:
- 한국어 + 영어 동시 지원 (FAQ는 한국어, 상품/리뷰는 영어)
- 무료 로컬 실행 가능
- 모델 크기 471MB로 속도/성능 균형

**검색 품질 테스트 결과**

| 쿼리 | 상위 결과 | 유사도 |
|------|----------|--------|
| "배송 며칠 걸려요?" | FAQ: 배송은 며칠 걸리나요? | 0.68 |
| "환불 정책이 어떻게 돼요?" | FAQ: 세일 상품도 환불 가능한가요? | 0.76 |
| "hiking boots waterproof" | Product: Men's Trail Model 4 Waterproof Hiking Boots | 0.75 |

**ChromaDB 설정**
```python
collection = client.create_collection(
    name="shopping_rag",
    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
)
```

---

### 3단계 — RAG 파이프라인 구축 (1주)

**LangChain RAG 체인 구성**
```
사용자 질문 → 임베딩 → ChromaDB 검색(k=5) → 프롬프트 조합 → LLM → 답변
```

**소스 필터링 전략**

리뷰 데이터(영어)가 한국어 FAQ 검색을 오염시키는 문제를 발견하여
질문 의도에 따라 검색 소스를 동적으로 필터링하는 방식을 적용했습니다.

```python
# 상품 관련 질문 → faq + product 검색
# 일반 질문 → faq만 검색
if is_product_query:
    allowed = {"product", "faq"}
else:
    allowed = {"faq"}
```

**프롬프트 엔지니어링**

초기에는 한국어로 지시문을 작성했으나, llama3.2가 한국어 지시를 불완전하게 따르는 현상 발견.
영어 지시문으로 변경 후 할루시네이션이 감소했습니다.

---

### 실험: Ollama(llama3.2) vs OpenAI(GPT-4o-mini) 비교

| 항목 | llama3.2 (Ollama) | GPT-4o-mini (OpenAI) |
|------|------------------|---------------------|
| 비용 | 무료 | ~$0.001/회 |
| 한국어 품질 | 간헐적 언어 혼용 | 안정적 |
| 할루시네이션 | 발생 (없는 상품 생성) | 거의 없음 |
| 응답 속도 | 로컬 GPU 의존 | 빠름 |
| 인터넷 필요 | 불필요 | 필요 |

**결론:** 프로덕션 환경에서는 GPT-4o-mini, 비용 제약 환경에서는 llama3.2 권장

---

## 설치 및 실행

```bash
# 1. 환경 설정
conda create -n rag_env python=3.11
conda activate rag_env
pip install sentence-transformers chromadb langchain langchain-community langchain-openai tqdm

# 2. 데이터 준비
python step1_data_prep.py

# 3. 벡터 DB 구축
python step2_embedding.py

# 4. RAG 파이프라인 실행
python step3_rag_pipeline.py
```

---

## 프로젝트 구조

```
LLM/
├── chunks.jsonl              # 전처리된 청크 데이터
├── faq.csv                   # FAQ 데이터 (80개)
├── chroma_db/                # 벡터 DB
├── step1_data_prep.py        # 데이터 수집/전처리
├── step2_embedding.py        # 임베딩 + ChromaDB 구축
├── step3_rag_pipeline.py     # RAG 파이프라인
└── step4_streamlit_app.py    # 챗봇 UI (개발 중)
```

---

## 배운 점 및 개선 방향

**배운 점**
- 데이터 품질이 검색 정확도에 직결됨 (리뷰 데이터 오염 문제 직접 경험)
- chunk_size/overlap 값이 검색 품질에 영향을 미침
- 프롬프트 언어(한국어 vs 영어)에 따라 LLM 응답 품질이 달라짐

**개선 방향**
- Streamlit UI 추가 (4단계)
- 리뷰 데이터 한국어 번역 후 재임베딩
- 청크 사이즈 실험 (300/500/700 비교)
- Re-ranking 기법 적용으로 검색 정확도 향상

---

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com)
- [ChromaDB 공식 문서](https://docs.trychroma.com)
- [Sentence Transformers](https://www.sbert.net)
- 데이터셋: Kaggle Amazon Reviews, Kaggle E-commerce Products 

-------------0411 1일차 프로젝트 결과 리드미
