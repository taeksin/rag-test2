import os
import streamlit as st
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 API 키 불러오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
    st.stop()

# 임베딩 모델 초기화 (API 키를 명시적으로 전달)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# 📌 "rag-test.txt"에서 텍스트 불러오기
def load_sidebar_text(filename="rag-test.txt"):
    if not os.path.exists(filename):
        st.error(f"파일 '{filename}'이 존재하지 않습니다.")
        return "파일을 찾을 수 없습니다."

    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()  # 파일 전체 내용을 읽음

    return content

# 미리 정의된 벡터 DB 텍스트
db_texts = [
    "세금",
    "소득",
    "납세",
    "과세",
    "조세",
    "부가세",
    "법인세",
    "관세",
    "신고",
    "공제",
    "감면",
    "소득세",
    "납부",
    "부역",
    "세목",
    "환급",
    "돈",
    "세금 깎기",
    "내야 할 돈",
    "돌려받을 돈",
    "혜택",
    "수입",
    "부과",
    "절세",
    "공납",
    "세율",
    "세금을 신고해야 한다.",
    "소득세를 납부하세요.",
    "세금을 내야 합니다.",
    "부가세를 납부하시오.",
    "세금을 줄이는 방법이 있다.",
    "세금 환급을 받을 수 있다.",
    "부역을 면제받았다.",
    "공납을 완료하였다.",
    "그는 조세를 납부하였다.",
    "세율이 변경되었다.",
    "근로소득세 신고 기한 내에 반드시 신고해야 한다.",
    "세금 신고 마감일이 다가오니 서둘러야 한다.",
    "절세를 위해 공제 항목을 꼼꼼히 확인하세요.",
    "세금을 내는 것이 국민의 의무이다."
    "그는 조세를 부담하며 국가에 충성하였다. "
]

# DB 텍스트를 임베딩하는 함수
def process_embeddings(texts):
    embeddings = embedding_model.embed_documents(texts)
    embeddings_array = np.array(embeddings)
    st.write(f"DB 텍스트 수: {len(texts)}")
    # st.write(f"임베딩 배열 shape: {embeddings_array.shape}")
    return embeddings_array

# 미리 DB 텍스트 임베딩 생성
db_embeddings = process_embeddings(db_texts)

# 3D 시각화 함수 (PCA)
def create_visualization_3d(embeddings_array, texts, query_embedding=None, query_text=None):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings_array)
    df = pd.DataFrame(reduced, columns=['X축', 'Y축', 'Z축'])
    df['text'] = texts
    df['text'] = df['text'].apply(lambda x: x[:5])  # 5글자로 제한
    
    fig = px.scatter_3d(df, x='X축', y='Y축', z='Z축', text='text',
                        title='DB 텍스트 임베딩 3D 시각화')
    fig.update_traces(textposition='top center', marker=dict(size=6))
    
    # 쿼리 임베딩 표시
    if query_embedding is not None and query_text is not None:
        query_reduced = pca.transform(query_embedding)
        fig.add_trace(
            go.Scatter3d(
                x=[query_reduced[0, 0]],
                y=[query_reduced[0, 1]],
                z=[query_reduced[0, 2]],
                mode='markers+text',
                marker=dict(size=8, color='red'),
                text=[query_text],
                name='Query'
            )
        )
    
    explained_variance = pca.explained_variance_ratio_
    st.write("3D PCA 각 주성분이 설명하는 분산 비율:")
    for i, ratio in enumerate(explained_variance):
        st.write(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    return fig, df

# 2D 시각화 함수 (PCA)
def create_visualization_2d(embeddings_array, texts, query_embedding=None, query_text=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings_array)
    df = pd.DataFrame(reduced, columns=['X축', 'Y축'])
    df['text'] = texts
    df['text'] = df['text'].apply(lambda x: x[:5])  # 5글자로 제한

    fig = px.scatter(df, x='X축', y='Y축', text='text',
                     title='DB 텍스트 임베딩 2D 시각화')
    fig.update_traces(textposition='top center', marker=dict(size=6))
    
    # 쿼리 임베딩 표시
    if query_embedding is not None and query_text is not None:
        query_reduced = pca.transform(query_embedding)
        fig.add_trace(
            go.Scatter(
                x=[query_reduced[0, 0]],
                y=[query_reduced[0, 1]],
                mode='markers+text',
                marker=dict(size=8, color='red'),
                text=[query_text],
                name='Query'
            )
        )
    
    explained_variance = pca.explained_variance_ratio_
    st.write("2D PCA 각 주성분이 설명하는 분산 비율:")
    for i, ratio in enumerate(explained_variance):
        st.write(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    return fig, df

# 단일 쿼리 검색: L2 거리와 코사인 유사도를 모두 계산
def search_query(query, db_embeddings, db_texts):
    query_embedding = embedding_model.embed_documents([query])
    query_embedding = np.array(query_embedding)
    # L2 거리 계산 (작을수록 유사)
    l2_distances = np.linalg.norm(db_embeddings - query_embedding[0], axis=1)
    # 코사인 유사도 계산 (1에 가까울수록 유사)
    cos_sim = cosine_similarity(db_embeddings, query_embedding)
    cos_sim = cos_sim.flatten()
    # 결과: (텍스트, L2 거리, 코사인 유사도)
    results = list(zip(db_texts, l2_distances, cos_sim))
    # L2 거리 기준 오름차순 정렬
    results_sorted = sorted(results, key=lambda x: x[1])
    return results_sorted, query_embedding

# Streamlit UI
st.title("임베딩 테스트_small")
st.write("검색할 텍스트를 입력하면, 미리 임베딩된 DB와의 L2 거리와 코사인 유사도 점수를 계산합니다.")

# 사용자에게 단일 텍스트 입력 받기
query_input = st.text_input("검색할 텍스트를 입력하세요:", placeholder="예: 관세")

if st.button("검색 실행"):
    if query_input:
        results, query_embedding = search_query(query_input, db_embeddings, db_texts)
        
        st.subheader("검색 결과")
        # 결과를 DataFrame으로 변환하여 출력 (L2 거리는 낮을수록, 코사인 유사도는 1에 가까울수록 유사)
        df_results = pd.DataFrame(results, columns=["텍스트", "L2 거리", "코사인 유사도"])
        st.dataframe(df_results.style.format({"L2 거리": "{:.4f}", "코사인 유사도": "{:.4f}"}))
        
        # 3D, 2D 시각화를 탭으로 표시
        tab1, tab2 = st.tabs(["3D 시각화", "2D 시각화"])
        with tab1:
            fig_3d, _ = create_visualization_3d(db_embeddings, db_texts, query_embedding=query_embedding, query_text=query_input)
            st.plotly_chart(fig_3d)
        with tab2:
            fig_2d, _ = create_visualization_2d(db_embeddings, db_texts, query_embedding=query_embedding, query_text=query_input)
            st.plotly_chart(fig_2d)
    else:
        st.warning("검색할 텍스트를 입력해주세요.")

with st.sidebar:
    st.header("사용 방법")
    st.write("""
    1. 검색할 텍스트를 입력합니다. (입력은 한 개만 받습니다.)
    2. 미리 임베딩된 벡터 DB에는 다음 데이터가 포함되어 있습니다:
    3. '검색 실행' 버튼을 클릭하면, 각 텍스트에 대해 L2 거리와 코사인 유사도 점수를 계산하여 결과를 테이블로 보여줍니다.
    4. 탭을 이용하여 3D와 2D 시각화를 모두 확인할 수 있습니다.
    5. 입력된 데이터
        """)
    sidebar_text = load_sidebar_text()
    st.write(sidebar_text)  # 파일 내용 출력
