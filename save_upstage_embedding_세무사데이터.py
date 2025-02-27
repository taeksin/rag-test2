import os
import pickle
import numpy as np
import pandas as pd
import faiss
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from openai import OpenAI  # Upstage API 사용

# .env 파일 로드
load_dotenv()

# Upstage API 키 설정
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    print("UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# Upstage API 클라이언트 초기화
client = OpenAI(
    api_key=api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

# 📌 엑셀 파일 목록 및 시트 지정
excel_files = [
    ("data_source/세무사 데이터전처리_20250116.xlsx", ["Sheet1"]),
]

# ✅ 모든 데이터를 하나의 리스트에 저장
documents = []

# 📌 엑셀 파일에서 데이터 읽어오기
for file, sheets in excel_files:
    for sheet in sheets:
        # 엑셀 데이터 로드
        df = pd.read_excel(file, engine='openpyxl', sheet_name=sheet)

        # NaN 데이터 제거
        df = df.dropna(subset=["제목", "본문_원본"])

        # 각 행을 Document로 변환하여 리스트에 추가
        for _, row in df.iterrows():
            content = f"제목: {row['제목']}\n본문: {row['본문_원본']}"
            source = f"{file.replace('data_source/', '')}__{sheet}"

            metadata = {
                "파일명": row.get("파일명", ""),
                "문서명": row.get("문서명", ""),
                "제목": row["제목"],
                "본문_원본": row["본문_원본"],
                "source": source
            }

            documents.append(Document(page_content=content, metadata=metadata))

# ✅ 데이터 개수 출력
print(f"✅ 엑셀에서 읽은 총 문서 개수: {len(documents)}")

# 📌 문서 내용 리스트 생성 (임베딩할 텍스트만 추출)
text_contents = [doc.page_content for doc in documents]

# 📌 Upstage API를 사용하여 임베딩 생성
def get_upstage_embedding(texts):
    """
    Upstage API를 사용하여 벡터 생성 (batch_size=100)
    """
    embeddings = []
    batch_size = 100  # Upstage API 최대 요청 개수
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model="embedding-passage",
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"🚨 임베딩 요청 실패: {str(e)}")
    return embeddings

# ✅ 임베딩 생성 시작
embeddings = get_upstage_embedding(text_contents)
print(len(embeddings))
if len(embeddings) != len(documents):
    print("🚨 임베딩 개수가 문서 개수와 일치하지 않습니다.")
    exit()

# 📌 FAISS 벡터 저장소 생성 (수정된 부분)
dimension = len(embeddings[0])  # Upstage 임베딩 벡터 차원
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

# 📌 FAISS 인덱스에 맞게 index_to_docstore_id 생성
index_to_docstore_id = {i: str(i) for i in range(len(documents))}
docstore = {str(i): doc for i, doc in enumerate(documents)}

vectorstore = FAISS(
    embedding_function=None,  # Upstage API를 사용하므로 필요 없음
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# 📌 FAISS 벡터 저장소 저장
SAVE_DIR = "vdb/upstage_faiss"
vectorstore.save_local(SAVE_DIR)

print("✅ FAISS 벡터 저장소 생성 완료!")
print(f"✅ FAISS 벡터 개수: {index.ntotal}")
print(f"✅ 저장된 문서 개수: {len(vectorstore.docstore)}")
