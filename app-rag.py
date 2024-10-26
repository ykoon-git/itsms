import streamlit as st
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
import json
from typing import List, Dict
from requests_aws4auth import AWS4Auth

# AWS 인증 설정
region = 'us-west-2'  # 예: 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

# OpenSearch Serverless 연결 설정
host = 'o0hj5d4vh1k6bxab969l.us-west-2.aoss.amazonaws.com'
port = 443

# OpenSearch 클라이언트 초기화
@st.cache_resource
def get_opensearch_client():
    return OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=awsauth,
        use_ssl=True,
        connection_class=RequestsHttpConnection
    )

# Bedrock 클라이언트 초기화
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=region
    )

def get_embedding(text: str, bedrock_client) -> List[float]:
    """Titan 임베딩 생성"""
    try:
        response = bedrock_client.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })
        )
        return json.loads(response['body'].read())['embedding']
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        return None

def similarity_search(query: str, k: int = 3, opensearch_client=None, bedrock_client=None) -> List[Dict]:
    """벡터 유사도 검색"""
    try:
        query_vector = get_embedding(query, bedrock_client)
        
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "vector_embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        
        results = opensearch_client.search(
            index="server_info",
            body=search_body
        )
        
        return [hit['_source'] for hit in results['hits']['hits']]
    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
        return []

def format_context(server_info: List[Dict]) -> str:
    """서버 정보를 문맥으로 포맷팅"""
    context = "다음은 관련된 서버 정보입니다:\n\n"
    for idx, info in enumerate(server_info, 1):
        context += f"서버 {idx}:\n"
        context += f"- 인스턴스: {info.get('instance_name', 'N/A')}\n"
        context += f"- 용도: {info.get('purpose', 'N/A')}\n"
        context += f"- 서비스: {info.get('service_name', 'N/A')}\n"
        context += f"- 상태: {info.get('server_status', 'N/A')}\n"
        context += f"- 사양: CPU {info.get('cpu', 'N/A')}코어, "
        context += f"메모리 {info.get('memory', 'N/A')}GB, "
        context += f"디스크 {info.get('disk', 'N/A')}GB\n"
        context += f"- OS: {info.get('os', 'N/A')}\n\n"
    return context

def generate_response(query: str, context: str, bedrock_client) -> str:
    try:
        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "messages": [
                {
                    "role": "user",
                    "content": f"""당신은 서버 인프라 전문가입니다. 
다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

{context}

사용자 질문: {query}

답변 시 다음 지침을 따라주세요:
1. 제공된 서버 정보만을 기반으로 답변하세요.
2. 정보가 없는 경우, 그 사실을 명시하고 알 수 있는 범위 내에서만 답변하세요.
3. 서버의 상세 정보를 언급할 때는 구체적인 수치와 함께 설명해주세요.
4. 전문적이고 정확한 용어를 사용하되, 이해하기 쉽게 설명해주세요."""
                }
            ]
        })

        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=request_body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        st.error(f"응답 생성 중 오류 발생: {str(e)}")
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("🖥️ 서버 인프라 검색 챗봇")
    
    initialize_session_state()
    
    # 클라이언트 초기화
    opensearch_client = get_opensearch_client()
    bedrock_client = get_bedrock_client()
    
    # 사이드바 설정
    with st.sidebar:
        st.header("검색 설정")
        k_value = st.slider("참조할 서버 수", min_value=1, max_value=5, value=3)
        
        st.markdown("""
        ### 사용 가이드
        1. 서버 관련 질문을 입력하세요
        2. 관련된 서버 정보를 검색합니다
        3. 검색된 정보를 바탕으로 답변을 생성합니다
        
        ### 질문 예시
        - 현재 운영 중인 서버의 상태는?
        - 파일 서버의 사양은?
        - Ubuntu 서버는 몇 대인가요?
        """)
    
    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력 처리
    if prompt := st.chat_input("서버 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 관련 서버 정보 검색
        with st.spinner("관련 정보를 검색 중입니다..."):
            relevant_servers = similarity_search(
                prompt, 
                k_value, 
                opensearch_client, 
                bedrock_client
            )
            context = format_context(relevant_servers)
        
        # 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있습니다..."):
                response = generate_response(prompt, context, bedrock_client)
                st.markdown(response)
        
        # 어시스턴트 메시지 저장
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 참조 정보 표시
        with st.expander("참조한 서버 정보"):
            st.json(relevant_servers)

if __name__ == "__main__":
    main()
