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
    "스님",
    "사람",
    "바람",
    "고양이",
    "강아지",
    "이름",
    "꽃",
    "달",
    "눈",
    "사랑",
    "눈",
    "달걀",
    "넋",
    "학문",
    "방",
    "수저",
    "나무",
    "길",
    "먹다",
    "버릇",
    "부족",
    "즐거움",
    "아픔",
    "사랑",
    "소망",
    "어둠",
    "꿈",
    "지혜",
    "향기",
    "말씀",
    "정서",
    "스님이 절에 간다",
    "사람이 길을 걷는다",
    "스님이 절로 향한다",
    "배가 고파 밥을 먹었다",
    "어머니께 편지를 썼다",
    "그는 책을 읽고 있었다",
    "학문을 닦다",
    "달이 휘영청 밝다",
    "그는 재물을 탐하였다",
    "그녀는 음식을 섭취하였다",
    "스님이 깊은 산속의 절에서 고요히 참선을 하고 있다.",
    "고요한 새벽, 산사에 울려 퍼지는 목탁 소리가 들린다",
    "스님은 깊은 산속 작은 절에서 홀로 수행을 하고 있었다",
    "그는 먼 길을 걸어와 친구를 만나 오랜만에 담소를 나누었다.",
    "봄바람이 불어오자 들판 가득 흐드러진 꽃이 피어나기 시작했다.",
    "오랜 가뭄 끝에 비가 내리자, 메말랐던 땅이 촉촉해졌다.",
    "그는 한숨을 내쉬며 창밖으로 흐린 하늘을 바라보았다.",
    "고즈넉한 산사에 종소리가 울려 퍼지며 새벽을 알렸다.",
    "그는 문득 학문의 깊음을 깨닫고 더욱 정진하기로 다짐했다.",
    "창가에 앉아 달빛을 바라보며 지난 세월을 곱씹었다",
    "이른 새벽, 먼동이 트기 전에 닭이 우는 소리가 들려왔다.",
    "그는 오랜 길을 걸어와 마침내 스승을 찾아뵈었다.",
    "이른 새벽부터 책을 펴고 배움에 몰두하고 있었다",
    "수풀이 우거진 길을 따라 걸으며 자연의 소리를 들었다."
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
        st.write(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
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
st.write("검색할 텍스트를 입력하면, 미리 임베딩된 DB(스님, 이름이름이름, 달걀, 넋, 버릇, 부족, 즐거움)와의 L2 거리와 코사인 유사도 점수를 계산합니다.")

# 사용자에게 단일 텍스트 입력 받기
query_input = st.text_input("검색할 텍스트를 입력하세요:", placeholder="예: 달걀")

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
    
    -1. 단일 단어 - 단일 단어
    
    -✅ 완전히 동일한 단어
    -스님 - 스님
    -사람 - 사람
    -바람 - 바람
    
    -✅ 고전 vs 현대 표현 차이 없음 (동일 단어이지만 표기만 다름)
    -고양이 - 고양이
    -강아지 - 강아지
    
    -2. 단일 단어 - 복수 단어
    
    -✅ 단어가 반복된 경우
    -스님 - 스님스님스님
    -이름 - 이름이름이름
    -꽃 - 꽃꽃꽃
    
    -✅ 한 개의 단어가 확장된 경우 (수식어 추가)
    -스님 - 젊은 스님
    -이름 - 그의 이름
    -고양이 - 검은 고양이
    
    -✅ 단어가 합쳐진 경우 (중첩 구조)
    -달 - 달빛
    -눈 - 눈물
    -바람 - 바람결
    
    -✅ 동일 개념이지만 단어가 결합된 경우
    -사랑 - 첫사랑
    -눈 - 눈송이
    -달 - 보름달
    
    -3. 같은 의미, 다른 표현의 단어
    
    -✅ 동의어 관계
    -스님 - 승려
    -달걀 - 계란
    -넋 - 혼
    
    -✅ 고전 표현 vs 현대 표현
    -학문 - 공부
    -방 - 거처
    -수저 - 식기
    
    -✅ 순우리말 vs 한자어
    -나무 - 목재
    -길 - 도로
    -먹다 - 섭취하다
    
    -4. 유사한 의미, 다른 표현의 단어
    
    -✅ 비슷한 개념이지만 뉘앙스가 다른 단어
    -버릇 - 습관
    -부족 - 결핍
    -즐거움 - 기쁨
    -아픔 - 고통
    -사랑 - 애정
    
    -✅ 정확히 같은 개념은 아니지만, 상황에 따라 비슷하게 쓰이는 단어
    -소망 - 희망
    -어둠 - 암흑
    -꿈 - 환상
    -지혜 - 지식
    
    -✅ 고전적 표현 vs 일반적 표현
    -향기 - 냄새
    -말씀 - 대화
    -정서 - 감정
    
    -5. 짧은 문장 - 짧은 문장 ( 6 ~ 15 자 이내)
    
    -✅ 동일 문장
    -스님이 절에 간다 - 스님이 절에 간다
    -사람이 길을 걷는다 - 사람이 길을 걷는다
    
    -✅ 같은 의미, 다른 표현
    -스님이 절로 향한다 - 승려가 사찰로 간다
    -배가 고파 밥을 먹었다 - 허기가 져서 식사를 했다
    
    -✅ 유사한 의미, 다소 다른 표현
    -어머니께 편지를 썼다 - 어머니께 글을 남겼다
    -그는 책을 읽고 있었다 - 그는 독서를 하고 있었다
    
    -✅ 고전적 표현 vs 현대적 표현
    -학문을 닦다 - 공부를 하다
    -달이 휘영청 밝다 - 달이 매우 밝다
    
    -✅ 한자어 포함 vs 순우리말
    -그는 재물을 탐하였다 - 그는 돈을 욕심냈다
    -그녀는 음식을 섭취하였다 - 그녀는 밥을 먹었다
    
    -6. 중간 문장 - 중간 문장 ( 30 ~ 60 자 이내 )
    
    -✅ 완전히 동일한 문장
    -스님이 깊은 산속의 절에서 고요히 참선을 하고 있다. - 스님이 깊은 산속의 절에서 고요히 참선을 하고 있다.
    -고요한 새벽, 산사에 울려 퍼지는 목탁 소리가 들린다. - 고요한 새벽, 산사에 울려 퍼지는 목탁 소리가 들린다.
    
    -✅ 같은 의미, 표현이 다름
    -스님은 깊은 산속 작은 절에서 홀로 수행을 하고 있었다. - 승려는 조용한 사찰에서 묵묵히 깨달음을 구하고 있었다.
    -그는 먼 길을 걸어와 친구를 만나 오랜만에 담소를 나누었다. - 오래 떨어져 지낸 친구와 길 끝에서 재회하여 이야기를 나누었다.
    -봄바람이 불어오자 들판 가득 흐드러진 꽃이 피어나기 시작했다. - 산들바람이 스치자 들녘에 핀 꽃들이 살랑이며 춤을 추었다.
    
    -✅ 비슷한 의미지만 뉘앙스가 다름
    -오랜 가뭄 끝에 비가 내리자, 메말랐던 땅이 촉촉해졌다. - 길었던 가뭄이 끝나고 단비가 내려, 들판이 생기를 되찾았다.
    -그는 한숨을 내쉬며 창밖으로 흐린 하늘을 바라보았다. - 창가에 앉아 흐린 하늘을 올려다보며 조용히 깊은 숨을 내쉬었다.
    -고즈넉한 산사에 종소리가 울려 퍼지며 새벽을 알렸다. - 조용한 절에 은은한 종소리가 퍼지며 새로운 하루가 시작되었다.
    
    -✅ 고전적 표현 vs 현대적 표현
    -그는 문득 학문의 깊음을 깨닫고 더욱 정진하기로 다짐했다. - "그는 공부의 어려움을 느끼고 더욱 노력하기로 결심했다.
    -창가에 앉아 달빛을 바라보며 지난 세월을 곱씹었다. - "창문 너머 밝은 달을 보며 지나온 날들을 되새겼다.
    -이른 새벽, 먼동이 트기 전에 닭이 우는 소리가 들려왔다. - "아침 해가 뜨기 전, 마당에서 닭 울음소리가 울려 퍼졌다.
    
    -✅ 순우리말 vs 한자어 표현
    -그는 오랜 길을 걸어와 마침내 스승을 찾아뵈었다. - 그는 장시간 도보한 끝에 사부를 알현하였다.
    -이른 새벽부터 책을 펴고 배움에 몰두하고 있었다. - 동이 트기도 전에 서책을 펼쳐 학문에 전념하고 있었다.
    -수풀이 우거진 길을 따라 걸으며 자연의 소리를 들었다. - 울창한 숲길을 거닐며 자연의 운율을 감상하였다.
        """)
