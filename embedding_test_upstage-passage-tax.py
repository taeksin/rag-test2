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

# FAISS 벡터 저장소 불러오기
from langchain_community.vectorstores import FAISS

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

# ── 쿼리 임베딩 생성 함수 ──
def get_query_embedding(text):
    try:
        response = client.embeddings.create(
            input=text, 
            model="embedding-passage"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"쿼리 임베딩 요청 실패: {str(e)}")
        return None

# ── 저장된 FAISS 벡터 저장소 로드 ──
try:
    vectorstore = FAISS.load_local(
        "vdb/upstage_faiss",
        embeddings=get_query_embedding,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"FAISS 벡터 저장소 로드 실패: {str(e)}")
    st.stop()

# 저장소에서 DB 문서와 임베딩 배열 추출
db_texts = [doc.page_content for doc in vectorstore.docstore.values()]
db_embeddings = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)

# ── 쿼리와 FAISS 벡터 저장소를 이용하여 전체 문서에 대해 유사도 검색하는 함수 ──
def search_query_vectorstore(query, k=None):
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        st.error("쿼리 임베딩 생성 실패")
        return [], None
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    # k 값이 지정되지 않으면 전체 문서를 대상으로 검색
    if k is None:
        k = vectorstore.index.ntotal

    # FAISS 인덱스를 사용해 k개의 가장 가까운 문서 검색 (L2 거리 기준)
    distances, indices = vectorstore.index.search(query_vector, k)
    
    # 코사인 유사도 계산 (전체 DB 임베딩에 대해)
    cos_sim_all = cosine_similarity(db_embeddings, query_vector).flatten()
    
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        doc_id = vectorstore.index_to_docstore_id[idx]
        doc = vectorstore.docstore[doc_id]
        cosine_value = cos_sim_all[idx]
        results.append((doc.page_content, distance, cosine_value))
    return results, query_vector

# ── 3D 시각화 함수 (PCA 이용) ──
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

# ── 2D 시각화 함수 (PCA 이용) ──
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
    st.title("Upstage DB(passage 임베딩) + Query 테스트(passage 임베딩)")
    st.write("DB에 담긴 데이터는 세무사 데이터전처리_20250116.xlsx 에서  [ 제목, 본문_원본 ] ")
    
    # 사이드바 구성: 사용 방법 안내
    with st.sidebar:
        st.header("사용 방법")
        st.write("""
        1. 검색할 텍스트를 입력합니다. (예: 관세)
        2. '검색 실행' 버튼 또는 'Enter' 키를 누르면 실행됩니다.
        3. 결과는 유사도가 높은 순으로 정렬되어 표시됩니다.
        4. 하단의 탭을 통해 2D와 3D 시각화를 확인할 수 있습니다.
        """)
    
    # 세션 상태 초기화
    if "query_trigger" not in st.session_state:
        st.session_state.query_trigger = False

    # 사용자 입력 받기
    query = st.text_input("검색할 텍스트를 입력하세요:", placeholder="예: 관세", key="query")

    # 엔터 입력 시 자동 실행
    if query and st.session_state.query_trigger is False:
        st.session_state.query_trigger = True

    # 버튼 또는 엔터 입력 시 실행
    if st.button("검색 실행") or st.session_state.query_trigger:
        if query:
            results, query_embedding = search_query_vectorstore(query)
            if results:
                # DataFrame 생성 후 "코사인 유사도" 기준 내림차순 정렬하여 모든 결과 표시
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
