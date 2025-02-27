import os
import pickle
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
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

# 📌 "rag-test.txt"에서 텍스트 불러오기
def load_sidebar_text(filename="rag-test.txt"):
    if not os.path.exists(filename):
        st.error(f"파일 '{filename}'이 존재하지 않습니다.")
        return "파일을 찾을 수 없습니다."
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()  # 파일 전체 내용을 읽음
    return content

# 미리 정의된 벡터 DB 텍스트 불러오기
def load_db_texts(filename="rag-test DB_set_01.txt"):
    if not os.path.exists(filename):
        st.error(f"파일 '{filename}'이 존재하지 않습니다.")
        return []
    with open(filename, "r", encoding="utf-8") as file:
        # 각 줄을 읽고, 양 끝 공백 제거 후 빈 줄 제외
        texts = [line.strip() for line in file.readlines() if line.strip()]
    return texts

# DB 텍스트 로드
db_texts = load_db_texts()
if not db_texts:
    st.error("DB 텍스트 파일이 비어 있습니다.")
    st.stop()

# 로컬에 저장된 DB 임베딩 데이터 로드 함수
def load_embeddings_from_file(filepath="vdb/openai/large/db_embeddings.pkl"):
    if not os.path.exists(filepath):
        st.error(f"임베딩 파일 '{filepath}'을 찾을 수 없습니다.")
        st.stop()
    with open(filepath, "rb") as f:
        embeddings = pickle.load(f)
    return np.array(embeddings)

# 미리 저장된 DB 임베딩 로드
db_embeddings = load_embeddings_from_file()

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

    fig = px.scatter(df, x='X축', y='Y축', text='text', title='DB 텍스트 임베딩 2D 시각화')
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
st.title("임베딩 테스트_large")
st.write("검색할 텍스트를 입력하면, 미리 임베딩된 DB와의 L2 거리와 코사인 유사도 점수를 계산합니다.")

# 사용자에게 단일 텍스트 입력 받기
query_input = st.text_input("검색할 텍스트를 입력하세요:", placeholder="예: 관세")

if st.button("검색 실행"):
    if query_input:
        # 입력된 텍스트를 임베딩 변환
        query_embedding = embedding_model.embed_documents([query_input])
        query_embedding = np.array(query_embedding)

        # DB 임베딩과 비교
        results, _ = search_query(query_input, db_embeddings, db_texts)

        st.subheader("임베딩 기반 검색 결과")
        df_results = pd.DataFrame(results, columns=["텍스트", "L2 거리", "코사인 유사도"])
        
        # 결과 출력 (유사도 높은 순으로 정렬)
        df_results = df_results.sort_values(by="코사인 유사도", ascending=False)
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
