"""
RAG 챗봇 1단계: 데이터 수집 및 전처리
========================================
수집 데이터:
  - faq.csv          : 쇼핑몰 FAQ 80개 (직접 작성)
  - productsclassified.csv : Kaggle 상품 데이터 247개
  - Reviews.csv      : Kaggle Amazon Reviews (568,454개 중 상위 300개 샘플링)

결과물:
  - chunks.jsonl     : 총 2,082개 청크 (중복 제거 후 1,784개)

실행 전 설치:
    pip install pandas langchain-text-splitters tqdm
"""

import re
import json
import pandas as pd
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 설정 ──────────────────────────────────────────
FAQ_PATH      = "faq.csv"
PRODUCT_PATH  = "productsclassified.csv"
REVIEW_PATH   = "Reviews.csv"
OUTPUT_PATH   = "chunks.jsonl"
REVIEW_SAMPLE = 300      # 리뷰 샘플링 수 (유용도 상위)
CHUNK_SIZE    = 500      # 청크 최대 길이
CHUNK_OVERLAP = 50       # 청크 겹침 길이
# ──────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)


# ── 텍스트 정제 ────────────────────────────────────
def clean_text(text: str) -> str:
    """HTML 태그 제거 및 공백 정리"""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)      # HTML 태그 제거
    text = re.sub(r'\s+', ' ', text).strip() # 공백 정리
    return text


# ── 1. FAQ 데이터 처리 ─────────────────────────────
def process_faq(path: str) -> list[dict]:
    """
    FAQ 데이터 청킹
    - Q&A 한 쌍을 하나의 청크로 처리 (분리하지 않음)
    - chunk_size보다 짧아서 그대로 1개 청크로 유지
    """
    df = pd.read_csv(path, encoding='utf-8-sig')
    chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FAQ 처리 중"):
        text = f"Q: {clean_text(row['question'])}\nA: {clean_text(row['answer'])}"
        chunks.append({
            "doc_id": str(row['id']),
            "text": text,
            "source": "faq",
            "category": str(row['category'])
        })

    print(f"FAQ 청크: {len(chunks)}개")
    return chunks


# ── 2. 상품 데이터 처리 ────────────────────────────
def process_products(path: str) -> list[dict]:
    """
    상품 데이터 청킹
    - 상품명 + 설명을 텍스트로 변환
    - 긴 설명은 RecursiveCharacterTextSplitter로 분할
    """
    df = pd.read_csv(path, encoding='utf-8')
    chunks = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="상품 처리 중"):
        name     = clean_text(row.get('name', ''))
        category = clean_text(row.get('Classification', ''))

        # description 컬럼에서 텍스트 추출
        desc_raw = str(row.get('description', ''))
        # 리스트 형태 문자열 파싱
        desc_raw = re.sub(r"[\[\]'\"]", '', desc_raw)
        desc = clean_text(desc_raw)[:1000]  # 최대 1000자

        if not name or not desc:
            continue

        text = f"상품명: {name} 카테고리: {category} 설명: {desc}"
        sub_chunks = splitter.split_text(text)

        for i, chunk in enumerate(sub_chunks):
            chunks.append({
                "doc_id": f"product_{idx:04d}_{i}",
                "text": chunk,
                "source": "product",
                "category": category
            })

    print(f"상품 청크: {len(chunks)}개")
    return chunks


# ── 3. 리뷰 데이터 처리 ───────────────────────────
def process_reviews(path: str, sample_n: int = REVIEW_SAMPLE) -> list[dict]:
    """
    리뷰 데이터 청킹
    - HelpfulnessNumerator 기준 상위 N개 샘플링
    - 영어 리뷰 → 상품 추천 검색에 활용
    """
    df = pd.read_csv(path, encoding='utf-8')

    # 유용도 상위 샘플링
    df = df.sort_values('HelpfulnessNumerator', ascending=False).head(sample_n)
    chunks = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="리뷰 처리 중"):
        product_id = str(row.get('ProductId', f'review_{idx}'))
        summary    = clean_text(row.get('Summary', ''))
        review     = clean_text(row.get('Text', ''))
        score      = row.get('Score', '')

        if not review:
            continue

        text = f"[리뷰] {summary} 평점: {score}/5 내용: {review}"
        sub_chunks = splitter.split_text(text)

        for i, chunk in enumerate(sub_chunks):
            chunks.append({
                "doc_id": f"review_{product_id}_{i}",
                "text": chunk,
                "source": "review",
                "category": "고객리뷰"
            })

    print(f"리뷰 청크: {len(chunks)}개")
    return chunks


# ── 4. 저장 ───────────────────────────────────────
def save_chunks(chunks: list[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"\n저장 완료: {path} ({len(chunks)}개 청크)")


# ── 메인 실행 ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("RAG 챗봇 1단계: 데이터 전처리 시작")
    print("=" * 50)

    all_chunks = []

    # FAQ
    faq_chunks = process_faq(FAQ_PATH)
    all_chunks.extend(faq_chunks)

    # 상품
    product_chunks = process_products(PRODUCT_PATH)
    all_chunks.extend(product_chunks)

    try:
        review_chunks = process_reviews(REVIEW_PATH, REVIEW_SAMPLE)
        all_chunks.extend(review_chunks)
    except FileNotFoundError:
        print("Reviews.csv 없음 → 리뷰 데이터 스킵")
        print("(Kaggle에서 Amazon Reviews 데이터셋 다운로드 후 재실행)")

    save_chunks(all_chunks, OUTPUT_PATH)

    print(f"\n총 청크: {len(all_chunks)}개")
    print(f"  - FAQ    : {len(faq_chunks)}개")
    print(f"  - 상품   : {len(product_chunks)}개")
    if 'review_chunks' in locals():
        print(f"  - 리뷰   : {len(review_chunks)}개")
