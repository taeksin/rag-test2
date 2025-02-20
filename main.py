import os
import streamlit as st
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.decomposition import PCA
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

# 미리 정의된 벡터 DB 텍스트
db_texts = [
    "스님",
    "이름이름이름",
    "달걀",
    "넋",
    "버릇",
    "부족",
    "즐거움"
]

# DB 텍스트를 임베딩하는 함수
def process_embeddings(texts):
    embeddings = embedding_model.embed_documents(texts)
    embeddings_array = np.array(embeddings)
    st.write(f"DB 텍스트 수: {len(texts)}")
    st.write(f"임베딩 배열 shape: {embeddings_array.shape}")
    return embeddings_array

# 미리 DB 텍스트 임베딩 생성
db_embeddings = process_embeddings(db_texts)

# 3D 시각화 함수 (PCA)
def create_visualization_3d(embeddings_array, texts, query_embedding=None, query_text=None):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings_array)
    df = pd.DataFrame(reduced, columns=['PC1', 'PC2', 'PC3'])
    df['text'] = texts

    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', text='text',
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
        st.write(f"PC{i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")
    return fig, df

# 2D 시각화 함수 (PCA)
def create_visualization_2d(embeddings_array, texts, query_embedding=None, query_text=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings_array)
    df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    df['text'] = texts

    fig = px.scatter(df, x='PC1', y='PC2', text='text',
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
        st.write(f"PC{i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")
    return fig, df

# 단일 쿼리 검색: 입력 텍스트 임베딩 후 DB 임베딩과 L2 거리 계산
def search_query(query, db_embeddings, db_texts):
    query_embedding = embedding_model.embed_documents([query])
    query_embedding = np.array(query_embedding)
    # L2 거리 계산 (작을수록 유사)
    distances = np.linalg.norm(db_embeddings - query_embedding[0], axis=1)
    results = list(zip(db_texts, distances))
    # L2 거리가 낮은 순서대로 정렬
    results_sorted = sorted(results, key=lambda x: x[1])
    return results_sorted, query_embedding

# Streamlit UI
st.title("벡터 DB 검색 및 시각화 (L2 거리 기준)")
st.write("검색할 텍스트를 입력하면, 미리 임베딩된 DB(스님, 이름이름이름, 달걀, 넋, 버릇, 부족, 즐거움)와의 L2 거리 기준으로 유사도를 계산합니다.")

# 사용자에게 단일 텍스트 입력 받기
query_input = st.text_input("검색할 텍스트를 입력하세요:", placeholder="예: 달걀")

if st.button("검색 실행"):
    if query_input:
        # 검색 수행
        results, query_embedding = search_query(query_input, db_embeddings, db_texts)
        
        st.subheader("검색 결과 (낮은 L2 거리 순)")
        for idx, (text, distance) in enumerate(results):
            st.write(f"{idx+1}. 텍스트: **{text}**, L2 거리: {distance:.4f}")
        
        # 탭을 이용하여 3D와 2D 시각화를 동시에 보여줌
        tab1, tab2 = st.tabs(["3D 시각화", "2D 시각화"])
        with tab1:
            fig_3d, _ = create_visualization_3d(db_embeddings, db_texts, query_embedding=query_embedding, query_text=query_input)
            st.plotly_chart(fig_3d)
        with tab2:
            fig_2d, _ = create_visualization_2d(db_embeddings, db_texts, query_embedding=query_embedding, query_text=query_input)
            st.plotly_chart(fig_2d)
    else:
        st.warning("검색할 텍스트를 입력해주세요.")

# 사이드바에 사용법 설명 추가
with st.sidebar:
    st.header("사용 방법")
    st.write("""
    1. 검색할 텍스트를 입력합니다. (입력은 한 개만 받습니다.)
    2. 미리 임베딩된 벡터 DB에는 다음 데이터가 포함되어 있습니다:
       - 스님  
       - 이름이름이름  
       - 달걀  
       - 넋  
       - 버릇  
       - 부족  
       - 즐거움
    3. '검색 실행' 버튼을 클릭하면, L2 거리를 기준으로 유사도를 계산합니다.
    4. 탭을 이용하여 3D와 2D 시각화를 모두 확인할 수 있습니다.
    """)
