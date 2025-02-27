import os
import pickle
import numpy as np
import pandas as pd
from openai import OpenAI  # pip install openai (버전 openai==1.52.2)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import streamlit as st

# .env 파일 로드
load_dotenv()

# .env에서 UPSTAGE API 키 불러오기
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    st.error("UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
    st.stop()

# Upstage API 클라이언트 초기화
client = OpenAI(
    api_key=api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

# DB 텍스트 로드 함수
def load_db_texts(filename="rag-test DB_set_02.txt"):
    if not os.path.exists(filename):
        st.error(f"파일 '{filename}'이 존재하지 않습니다.")
        return []
    with open(filename, "r", encoding="utf-8") as file:
        texts = [line.strip() for line in file.readlines() if line.strip()]
    return texts

# 사이드바 텍스트 로드 함수
def load_sidebar_text(filename="rag-test.txt"):
    if not os.path.exists(filename):
        st.error(f"파일 '{filename}'이 존재하지 않습니다.")
        return "파일을 찾을 수 없습니다."
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    return content

# 로컬에 저장된 임베딩 데이터 불러오기
def load_embeddings_from_file(filename="vdb/upstage/db_embeddings_set_02.pkl"):
    if not os.path.exists(filename):
        st.error(f"임베딩 파일 '{filename}'을 찾을 수 없습니다.")
        return None
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return np.array(embeddings)

# Upstage API를 사용하여 쿼리 임베딩을 가져오는 함수
def get_query_embedding(text):
    try:
        response = client.embeddings.create(
            input=text, 
            model="embedding-query"
            # model="embedding-passage"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"쿼리 임베딩 요청 실패: {str(e)}")
        return None

# 쿼리와 DB 임베딩 간 유사도 계산 함수
def search_query(query, db_embeddings, db_texts):
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        st.error("쿼리 임베딩 생성 실패")
        return [], None
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # L2 거리 계산 (값이 작을수록 유사)
    l2_distances = np.linalg.norm(db_embeddings - query_embedding, axis=1)
    # 코사인 유사도 계산 (1에 가까울수록 유사)
    cos_sim = cosine_similarity(db_embeddings, query_embedding).flatten()
    
    results = list(zip(db_texts, l2_distances, cos_sim))
    results_sorted = sorted(results, key=lambda x: x[1])
    return results_sorted, query_embedding

# 3D 시각화 함수 (PCA 이용)
def create_visualization_3d(embeddings_array, texts, query_embedding=None, query_text=None):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings_array)
    df = pd.DataFrame(reduced, columns=['X축', 'Y축', 'Z축'])
    df['text'] = texts
    df['text'] = df['text'].apply(lambda x: x[:5])  # 텍스트를 5글자로 제한
    
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

# 2D 시각화 함수 (PCA 이용)
def create_visualization_2d(embeddings_array, texts, query_embedding=None, query_text=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings_array)
    df = pd.DataFrame(reduced, columns=['X축', 'Y축'])
    df['text'] = texts
    df['text'] = df['text'].apply(lambda x: x[:5])  # 텍스트를 5글자로 제한

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

def main():
    st.title("Upstage DB(passage로 임베딩) + query(query로 임베딩) 테스트")
    st.write("쿼리 텍스트를 입력하면 로컬에 저장된 DB 임베딩과 비교하여 유사도를 계산하고, 2D 및 3D 시각화를 제공합니다.")
    
    # 사이드바 구성: 사용 방법 및 DB 텍스트 목록 표시
    with st.sidebar:
        st.header("사용 방법")
        st.write("""
        1. 검색할 텍스트를 입력합니다. (예: 관세)
        2. '검색 실행' 버튼 또는 'Enter' 키를 누르면 실행됩니다.
        3. 결과는 유사도가 높은 순으로 정렬되어 표시됩니다.
        4. 하단의 탭을 통해 2D와 3D 시각화를 확인할 수 있습니다.
        """)
        sidebar_text = load_sidebar_text()
        st.write(sidebar_text)
    
    # DB 텍스트 및 임베딩 불러오기
    db_texts = load_db_texts()
    if not db_texts:
        st.error("DB 텍스트 파일이 비어 있습니다.")
        st.stop()
    db_embeddings = load_embeddings_from_file()
    if db_embeddings is None:
        st.error("DB 임베딩 파일을 불러올 수 없습니다.")
        st.stop()
    
    # 세션 상태 초기화
    if "query_trigger" not in st.session_state:
        st.session_state.query_trigger = False

    # 사용자 입력 받기
    query = st.text_input("검색할 텍스트를 입력하세요:", placeholder="예: 관세", key="query")

    # 엔터 입력 시 자동 실행
    if query and st.session_state.query_trigger is False:
        st.session_state.query_trigger = True

    # 버튼을 누르거나 Enter 입력 시 실행
    if st.button("검색 실행") or st.session_state.query_trigger:
        if query:
            results, query_embedding = search_query(query, db_embeddings, db_texts)
            if results:
                df_results = pd.DataFrame(results, columns=["텍스트", "L2 거리", "코사인 유사도"])
                df_results = df_results.sort_values(by="코사인 유사도", ascending=False)
                st.subheader("검색 결과")
                st.dataframe(df_results.style.format({"L2 거리": "{:.4f}", "코사인 유사도": "{:.4f}"}))
                
                # 3D, 2D 시각화 탭 구성
                tab1, tab2 = st.tabs(["3D 시각화", "2D 시각화"])
                with tab1:
                    fig_3d, _ = create_visualization_3d(db_embeddings, db_texts, query_embedding=query_embedding, query_text=query)
                    st.plotly_chart(fig_3d)
                with tab2:
                    fig_2d, _ = create_visualization_2d(db_embeddings, db_texts, query_embedding=query_embedding, query_text=query)
                    st.plotly_chart(fig_2d)
            
            # 실행 후 상태 초기화
            st.session_state.query_trigger = False

if __name__ == "__main__":
    main()
