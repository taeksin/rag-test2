import os
import pickle
import numpy as np
from openai import OpenAI  # pip install openai (버전 openai==1.52.2)
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 UPSTAGE API 키 불러오기
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    print("UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# Upstage API 클라이언트 초기화
client = OpenAI(
    api_key=api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

# DB 텍스트 로드 함수
def load_db_texts(filename="rag-test DB용.txt"):
    if not os.path.exists(filename):
        print(f"파일 '{filename}'이 존재하지 않습니다.")
        return []
    with open(filename, "r", encoding="utf-8") as file:
        texts = [line.strip() for line in file.readlines() if line.strip()]
    return texts

# Upstage API를 사용하여 DB용 텍스트 임베딩을 배치로 가져오는 함수
def get_db_embedding(texts):
    """
    texts: DB 텍스트 리스트
    하나의 요청에는 최대 100개의 텍스트를 포함할 수 있으며,
    각 요청당 총 토큰 수가 204,800보다 작아야 합니다.
    """
    embeddings = []
    batch_size = 100  # 한 번에 최대 100개씩 요청
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model="embedding-passage",
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"임베딩 요청 실패: {str(e)}")
    return embeddings

# 저장할 디렉토리 설정
SAVE_DIR = "vdb/upstage"
SAVE_PATH = os.path.join(SAVE_DIR, "db_embeddings.pkl")

# 디렉토리 존재 여부 확인 후 생성
os.makedirs(SAVE_DIR, exist_ok=True)

# 임베딩 데이터를 로컬 파일에 저장하는 함수
def save_embeddings_to_file(embeddings, filename=SAVE_PATH):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)

def main():
    db_texts = load_db_texts()
    if not db_texts:
        print("DB 텍스트 파일이 비어 있습니다.")
    print(f"DB 텍스트 수: {len(db_texts)}")
    
    # DB 임베딩 생성
    db_embeddings = get_db_embedding(db_texts)
    print("DB 임베딩 생성 완료.")
    
    # 로컬에 저장
    save_embeddings_to_file(db_embeddings)
    print("임베딩 데이터를 'db_embeddings.pkl'에 저장했습니다.")

if __name__ == "__main__":
    main()
