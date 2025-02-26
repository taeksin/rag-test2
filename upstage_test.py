
import os
from openai import OpenAI # openai==1.52.2
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv()

# .env에서 API 키 불러오기
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    print("UPSTAGE_API_KEY .env 파일에 설정되어 있지 않습니다.")
    
client = OpenAI(
    api_key = api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

response = client.embeddings.create(
    input="Solar embeddings are awesome",
    model="embedding-query"
)

print(response.data[0].embedding)