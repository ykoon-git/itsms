import streamlit as st
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# AWS 및 OpenSearch 설정
region = 'us-west-2'  # 예: 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

# OpenSearch 연결 설정
host = 'o0hj5d4vh1k6bxab969l.us-west-2.aoss.amazonaws.com'  # 예: 'https://xxx.us-west-2.aoss.amazonaws.com'
port = 443

# OpenSearch 클라이언트 생성
opensearch_client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Bedrock 클라이언트 생성
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=region
)

def generate_opensearch_query(natural_language_query):
    schema_info = """
    Index Name: 
        Purpose : server_info (It is an index that stores server information and has the fields below.
        Fields:
        - instance_name (keyword): Server instance name (e.g., srv-1234)
        - cpu (interger): number of CPU logical core information (e.g., "4", "8")
        - memory (integer): Memory amount information (e.g., "16", "32")
        - disk (integer): Disk amount information (e.g., "500", "1000")
        - os (keyword): Operating system (e.g., "Ubuntu 20.04", "CentOS 7")
        - purpose (keyword): Server purpose (e.g., "Web Server", "Database Server")
        - service_name (keyword): Name of the service running on the server
        - ip_address (ip): IP address of the server
        - location (text): Physical location of the server
        - department (keyword): Department responsible for the server
        - last_updated (date): Date of last update
        - registration_date (date): Date when the server was registered
        - server_status (keyword): Current status of the server (e.g., "running", "shutdown", "stop")
    
    Index Name: web_log 
        Purpose : It is an index that stores web log information and has the fields below.
        Fields:
        - timestamp (date): Log entry creation time
        - ip_address (ip): Client's IP address
        - method (keyword): HTTP request method (e.g., "GET", "POST", "PUT", "DELETE")
        - url (text): Requested URL path
        - status_code (integer): HTTP response status code
        - user_agent (text): Client's User-Agent string
        - referrer (text): Request's Referrer URL
        - response_time (float): Time spent processing the request (seconds)
        - bytes_sent (long): Number of bytes sent in response
        - vector_embedding (knn_vector): Vector representation for machine learning tasks
    """

    messages = [
        {
            "role": "user",
            "content": f"Given the following schema information:\n\n{schema_info}\n\nGenerate an OpenSearch query in JSON format for the following natural language query. The query should use the appropriate fields and query types based on the schema. Natural language query: {natural_language_query}"
        }
    ]

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.0,
        "messages": messages
    })

    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        generated_query = response_body['content'][0]['text']
        
        # 생성된 쿼리에서 JSON 부분만 추출
        json_start = generated_query.find('{')
        json_end = generated_query.rfind('}') + 1
        json_query = generated_query[json_start:json_end]
        
        return json.loads(json_query)
    except Exception as e:
        st.error(f"Error in generate_opensearch_query: {str(e)}")
        return None


def get_opensearch_indices(client):
    indices = client.cat.indices(format="json")
    return [index['index'] for index in indices if not index['index'].startswith('.')]


def search_opensearch(query, index_name):
    try:
        response = opensearch_client.search(
            index=index_name,
            body=query
        )
        return response['hits']['hits']
    except Exception as e:
        st.error(f"Error in search_opensearch: {str(e)}")
        return []

# Streamlit 앱
st.title("Server Info Chatbot")

st.sidebar.header("Settings")
indices = get_opensearch_indices(opensearch_client)

selected_index = st.sidebar.selectbox(
    "사용할 Index를 선택하세요.",
    options=indices,
    index=0  # 기본값으로 첫 번째 인덱스 선택
)

# 사용자 입력
user_query = st.text_input("서버의 정보를 알려드립니다. 무엇이든 물어보세요.")

if user_query:
    # OpenSearch 쿼리 생성
    opensearch_query = generate_opensearch_query(user_query)
    
    if opensearch_query:
        st.write("Generated OpenSearch Query:")
        st.json(opensearch_query)
        
        # OpenSearch 검색 수행
        st.write(f"Searching index: {selected_index}")
        search_results = search_opensearch(opensearch_query, selected_index)
        
        if search_results:
            st.write(f"Found {len(search_results)} results:")
            for hit in search_results:
                st.json(hit['_source'])
        else:
            st.write("No results found.")
    else:
        st.write("Failed to generate OpenSearch query.")
