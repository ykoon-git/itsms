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

def search_opensearch(query):
    try:
        response = opensearch_client.search(
            index="server_info",
            body=query
        )
        return response['hits']['hits']
    except Exception as e:
        st.error(f"Error in search_opensearch: {str(e)}")
        return []

# Streamlit 앱
st.title("OpenSearch Query Executor")

# JSON 입력을 위한 텍스트 영역
query_json = st.text_area("Enter your OpenSearch Query JSON here:", height=200)

if query_json:
    try:
        # JSON 파싱
        query = json.loads(query_json)
        st.write("Parsed Query:")
        st.json(query)

        # 쿼리 실행 버튼
        if st.button("Execute Query"):
            # OpenSearch 검색 수행
            search_results = search_opensearch(query)
            
            if search_results:
                st.write(f"Found {len(search_results)} results:")
                for hit in search_results:
                    st.json(hit['_source'])
            else:
                st.write("No results found.")
    except json.JSONDecodeError:
        st.error("Invalid JSON. Please enter a valid JSON query.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.write("Please enter an OpenSearch Query JSON.")

# 쿼리 예시 표시
st.sidebar.title("Query Example")
example_query = {
    "query": {
        "match": {
            "os": "Ubuntu"
        }
    }
}
st.sidebar.json(example_query)
st.sidebar.write("This example query will find all servers running Ubuntu.")

# 사용 방법 안내
st.sidebar.title("How to Use")
st.sidebar.write("""
1. Enter your OpenSearch query JSON in the text area.
2. The app will parse and display the query.
3. Click 'Execute Query' to run the query.
4. Results will be displayed below the button.
""")
