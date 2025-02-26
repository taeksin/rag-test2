import os
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 API 키 불러오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
    exit(1)

# 사용할 임베딩모델 정의
# MODEL = "large"
MODEL = "small"


# 임베딩 모델 초기화 (text-embedding-3-{MODEL} 사용)
embedding_model = OpenAIEmbeddings(model=f"text-embedding-3-{MODEL}", openai_api_key=api_key)

# DB 텍스트 로드 함수
def load_db_texts(filename="rag-test DB용.txt"):
    if not os.path.exists(filename):
        print(f"ERROR: 파일 '{filename}'이 존재하지 않습니다.")
        exit(1)
    with open(filename, "r", encoding="utf-8") as file:
        texts = [line.strip() for line in file.readlines() if line.strip()]
    return texts

# 임베딩 데이터를 로컬 파일에 저장하는 함수
def save_embeddings_to_file(embeddings, save_dir=f"vdb/openai/{MODEL}", filename="db_embeddings.pkl"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"임베딩 데이터를 '{save_path}'에 저장했습니다.")

def main():
    # DB 텍스트 로드
    db_texts = load_db_texts()
    print(f"DB 텍스트 수: {len(db_texts)}")

    # DB 텍스트 임베딩 생성
    embeddings = embedding_model.embed_documents(db_texts)
    embeddings_array = np.array(embeddings)
    print("DB 임베딩 생성 완료.")

    # 임베딩 데이터 저장
    save_embeddings_to_file(embeddings_array)

if __name__ == "__main__":
    main()
