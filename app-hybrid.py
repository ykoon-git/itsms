import streamlit as st
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# 세션 상태 초기화
if 'expander_state' not in st.session_state:
    st.session_state.expander_state = False  # 기본값을 True로 설정
    
    
# 페이지 설정
st.set_page_config(
    page_title="서버 인프라 검색 서비스",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS Bedrock 클라이언트 초기화
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2'
    )

# AWS 인증 설정
region = 'us-west-2'  # 예: 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

# OpenSearch Serverless 연결 설정
host = 'o0hj5d4vh1k6bxab969l.us-west-2.aoss.amazonaws.com'
port = 443

# OpenSearch 클라이언트 생성
opensearch_client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Titan 임베딩 생성
def get_titan_embedding(text, bedrock_client, normalize=True):
    response = bedrock_client.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": normalize
        })
    )
    return json.loads(response['body'].read())['embedding']

# 사용할 수 있는 Index 가져오기
def get_opensearch_indices(client):
    indices = client.cat.indices(format="json")
    return [index['index'] for index in indices if not index['index'].startswith('.')]

# 앱 제목
st.title("🔍 서버 인프라 검색 시스템")
st.markdown("""
이 시스템은 Amazon Titan Embeddings V2와 OpenSearch를 활용한 하이브리드 검색 시스템입니다.
키워드 기반 검색과 의미 기반 검색을 결합하여 더 정확한 검색 결과를 제공합니다.
""")

# 사이드바 필터
st.sidebar.header("검색옵션")
indices = get_opensearch_indices(opensearch_client)

selected_index = st.sidebar.selectbox(
    "사용할 Index를 선택하세요.",
    options=indices,
    index=2  # 기본값으로 첫 번째 인덱스 선택
)

# 필터 설정
with st.sidebar.expander("상세 필터", expanded=True):
    os_filter = st.multiselect(
        "운영체제",
        ["Ubuntu 22.04", "CentOS 7", "Windows Server 2019", "Red Hat 8", "Amazon Linux 2"]
    )
    
    status_filter = st.multiselect(
        "서버 상태",
        ["running", "stopped", "maintenance", "terminated"]
    )
    
    cpu_range = st.slider(
        "CPU 코어 수",
        min_value=1,
        max_value=64,
        value=(1, 64)
    )
    
    memory_range = st.slider(
        "메모리 (GB)",
        min_value=2,
        max_value=256,
        value=(2, 256)
    )

    applyfilter = st.checkbox(
        label = "필터 적용",
        value=False
    )
    
    
# 검색 가중치 설정
with st.sidebar.expander("검색 가중치 설정", expanded=True):
    keyword_weight = st.slider(
        "키워드 검색 가중치",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )
    vector_weight = 1 - keyword_weight
    
    

# 메인 검색 인터페이스
search_query = st.text_input("검색어를 입력하세요", placeholder="예: database server, 웹서버")

if search_query:
    try:
        bedrock_client = get_bedrock_client()
        # opensearch_client = get_opensearch_client()
        
        # 쿼리 벡터 생성
        query_vector = get_titan_embedding(search_query, bedrock_client)
        
        # 검색 쿼리 구성
        search_body = {
            "size": 10,
            "track_scores": True,
            "query": {
                "bool": {
                    "should": [
                        # 텍스트 검색
                        {
                            "match": {
                                "full_text": {
                                    "query": search_query,
                                    "boost": keyword_weight  # 텍스트 검색 가중치
                                }
                            }
                        },
                        # 벡터 검색
                        {
                            "knn": {
                                "vector_embedding": {
                                    "vector": query_vector,  # Titan Embeddings로 생성한 벡터
                                    "k": 10,
                                    "boost": vector_weight
                                }
                            }
                        }
                    ]
                    # "minimum_should_match": 1
                    # 필터 조건 추가
                }
            }
        }
        
        # 필터 적용
        if applyfilter:
            if os_filter:
                search_body["query"]["bool"]["filter"].append({"terms": {"os": os_filter}})
            if status_filter:
                search_body["query"]["bool"]["filter"].append({"terms": {"server_status": status_filter}})
            
            if cpu_range:
                if "filter" not in search_body["query"]["bool"]:
                    search_body["query"]["bool"]["filter"] = []
                search_body["query"]["bool"]["filter"].append({
                    "range": {"cpu": {"gte": cpu_range[0], "lte": cpu_range[1]}}
                })

            if memory_range:
                if "filter" not in search_body["query"]["bool"]:
                    search_body["query"]["bool"]["filter"] = []
                search_body["query"]["bool"]["filter"].append({
                    "range": {"memory": {"gte": memory_range[0], "lte": memory_range[1]}}
                })
        
        # 검색 실행
        results = opensearch_client.search(
            index=selected_index,
            body=search_body
        )
        
        st.write(search_body)
        
        # 결과 처리
        hits = results['hits']['hits']
        st.subheader(f"검색 결과: {len(hits)}개 발견")
        
        # 결과를 데이터프레임으로 변환
        if hits:
            df = pd.DataFrame([hit['_source'] for hit in hits])
            
            # 통계 대시보드
            col1, col2 = st.columns(2)
            
            with col1:
                # OS 분포 차트
                os_counts = df['os'].value_counts()
                fig1 = px.pie(values=os_counts.values, names=os_counts.index, title='운영체제 분포')
                st.plotly_chart(fig1)
            
            with col2:
                # 서버 상태 분포
                status_counts = df['server_status'].value_counts()
                fig2 = px.bar(x=status_counts.index, y=status_counts.values, title='서버 상태 분포')
                st.plotly_chart(fig2)
            
            # 상세 결과 표시
            for hit in hits:
                with st.expander(f"🖥️ {hit['_source']['instance_name']} (스코어: {hit['_score']:.2f})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📋 기본 정보**")
                        st.write(f"🔹 OS: {hit['_source']['os']}")
                        st.write(f"🔹 상태: {hit['_source']['server_status']}")
                        st.write(f"🔹 위치: {hit['_source']['location']}")
                        st.write(f"🔹 부서: {hit['_source']['department']}")
                    
                    with col2:
                        st.markdown("**💻 리소스 정보**")
                        st.write(f"🔹 CPU: {hit['_source']['cpu']} cores")
                        st.write(f"🔹 메모리: {hit['_source']['memory']} GB")
                        st.write(f"🔹 디스크: {hit['_source']['disk']} GB")
                        st.write(f"🔹 IP: {hit['_source']['ip_address']}")
                    
                    st.markdown("**🎯 용도**")
                    st.write(hit['_source']['purpose'])
                    
                    st.markdown("**📅 날짜 정보**")
                    st.write(f"등록일: {hit['_source']['registration_date']}")
                    st.write(f"최종 수정일: {hit['_source']['last_updated']}")
                    
                    st.markdown("---")
                    st.write(hit['_source']['full_text'])
        else:
            st.warning("검색 결과가 없습니다.")
            
    except Exception as e:
        st.error(f"검색 중 오류가 발생했습니다: {str(e)}")

# 사용 가이드
with st.sidebar.expander("💡 사용 가이드"):
    st.markdown("""
    1. **검색어 입력**
        - 자연어로 검색 가능
        - 구체적인 키워드 사용 가능
    
    2. **필터 사용**
        - OS 선택
        - 서버 상태 선택
        - CPU/메모리 범위 지정
    
    3. **검색 가중치 조정**
        - 키워드 검색과 의미 기반 검색의 비중 조절 가능
    
    4. **결과 확인**
        - 통계 차트로 전체 현황 파악
        - 상세 정보는 확장 패널에서 확인
    """)
