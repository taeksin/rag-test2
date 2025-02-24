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
st.title("벡터 DB 검색 및 시각화 (L2 & Cosine Similarity)")
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
    
    #1. 단일 단어 - 단일 단어
    ✅ 완전히 동일한 단어

    세금 - 세금
    소득 - 소득
    납세 - 납세
    과세 - 과세

    ✅ 고전 vs 현대 표현 차이 없음 (동일 단어이지만 표기만 다름)

    조세 - 조세
    부가세 - 부가세
    법인세 - 법인세
    관세 - 관세

    #2. 단일 단어 - 복수 단어
    ✅ 단어가 반복된 경우

    납세 - 납세납세납세
    세금 - 세금세금세금
    과세 - 과세과세과세
    신고 - 신고신고신고

    ✅ 한 개의 단어가 확장된 경우 (수식어 추가)

    세금 - 부가가치세
    소득 - 종합소득
    신고 - 법인세 신고
    공제 - 세액공제

    ✅ 단어가 합쳐진 경우 (중첩 구조)

    조세 - 조세제도
    신고 - 신고절차
    공제 - 공제항목
    감면 - 감면혜택

    ✅ 동일 개념이지만 단어가 결합된 경우

    소득세 - 종합소득세
    세금 - 납부세금
    감면 - 세액감면
    신고 - 전자신고

    #3. 같은 의미, 다른 표현의 단어
    ✅ 동의어 관계

    세금 - 조세
    납부 - 지급
    신고 - 제출
    감면 - 공제

    ✅ 고전 표현 vs 현대 표현

    부역 - 세금
    세목 - 세금항목
    관세 - 수입세
    환급 - 돌려받음

    ✅ 순우리말 vs 한자어

    돈 - 세금
    세금 깎기 - 감면
    내야 할 돈 - 납부액
    돌려받을 돈 - 환급금

    #4. 유사한 의미, 다른 표현의 단어
    ✅ 비슷한 개념이지만 뉘앙스가 다른 단어

    혜택 - 감면
    수입 - 소득
    부과 - 징수
    절세 - 공제

    ✅ 정확히 같은 개념은 아니지만, 상황에 따라 비슷하게 쓰이는 단어

    소득 - 수익
    납부 - 지불
    세금 - 부담금
    신고 - 신청

    ✅ 고전적 표현 vs 일반적 표현

    부역 - 세금 납부
    조세 - 세금
    공납 - 납세
    세율 - 세금 비율

    #5. 짧은 문장 - 짧은 문장 (6 ~ 15 자 이내)
    ✅ 동일 문장

    세금을 신고해야 한다. - 세금을 신고해야 한다.
    소득세를 납부하세요. - 소득세를 납부하세요.

    ✅ 같은 의미, 다른 표현

    세금을 내야 합니다. - 조세를 부담해야 합니다.
    부가세를 납부하시오. - 부가가치세를 내십시오.

    ✅ 유사한 의미, 다소 다른 표현

    세금을 줄이는 방법이 있다. - 절세 전략이 존재한다.
    세금 환급을 받을 수 있다. - 공제 혜택을 받을 수 있다.

    ✅ 고전적 표현 vs 현대적 표현

    부역을 면제받았다. - 세금 감면을 받았다.
    공납을 완료하였다. - 납세를 마쳤다.

    ✅ 한자어 포함 vs 순우리말

    그는 조세를 납부하였다. - 그는 세금을 냈다.
    세율이 변경되었다. - 세금 비율이 달라졌다.

    #6. 중간 문장 - 중간 문장 (30 ~ 60 자 이내)
    ✅ 완전히 동일한 문장

    근로소득세 신고 기한 내에 반드시 신고해야 한다. - 근로소득세 신고 기한 내에 반드시 신고해야 한다.

    ✅ 같은 의미, 표현이 다름

    세금 신고 마감일이 다가오니 서둘러야 한다. - 조세 신고 기한이 임박했으니 신속하게 제출해야 한다.

    ✅ 비슷한 의미지만 뉘앙스가 다름

    절세를 위해 공제 항목을 꼼꼼히 확인하세요. - 세금을 줄이려면 감면 혜택을 충분히 활용하세요.

    ✅ 고전적 표현 vs 현대적 표현

    그는 조세를 부담하며 국가에 충성하였다. - 그는 세금을 납부하며 국가의 의무를 다했다.

    ✅ 순우리말 vs 한자어 표현

    세금을 내는 것이 국민의 의무이다. - 조세 납부는 국민의 책무이다.
        """)
