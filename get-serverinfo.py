import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# AWS 설정
region = 'us-west-2'  # 예: 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

# OpenSearch Serverless 연결 설정
host = 'o0hj5d4vh1k6bxab969l.us-west-2.aoss.amazonaws.com'  # 실제 엔드포인트로 변경하세요
port = 443

# OpenSearch 클라이언트 생성
opensearch_client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Amazon Bedrock 클라이언트 생성
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
    aws_access_key_id=credentials.access_key,
    aws_secret_access_key=credentials.secret_key,
    aws_session_token=credentials.token
)

def generate_opensearch_query(natural_language_query):
    schema_info = """
    Index Name: server_info
    Fields:
    - instance_name (text): Server instance name (e.g., srv-1234)
    - cpu (integer): CPU information (e.g., "4", "8")
    - memory (integer): Memory information (e.g., "16", "32")
    - disk (integer): Disk information (e.g., "500", "1000")
    - os (text): Operating system (e.g., "Ubuntu 20.04", "CentOS 7")
    - purpose (text): Server purpose (e.g., "Web Server", "Database Server")
    - service_name (text): Name of the service running on the server
    - ip_address (ip): IP address of the server
    - location (text): Physical location of the server
    - department (text): Department responsible for the server
    - last_updated (date): Date of last update
    - registration_date (date): Date when the server was registered
    - server_status (text): Current status of the server (e.g., "running", "shutdown", "stop")

    Index Name: weblog_info
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
        - dimension: 1024
        - method:
            - name: hnsw
            - space_type: l2

    """

    messages = [
        {
            "role": "user",
            "content": f"Given the following schema information:\n\n{schema_info}\n\nGenerate an OpenSearch query in JSON format for the following natural language query. The query should use the appropriate fields and query types based on the schema. Natural language query: {natural_language_query}"
        }
    ]

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "temperature": 0.3,
        "top_k": 250, 
        "top_p": 0.5, 
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
        
        print(generated_query)
        
        # 생성된 쿼리에서 JSON 부분만 추출
        json_start = generated_query.find('{')
        json_end = generated_query.rfind('}') + 1
        json_query = generated_query[json_start:json_end]
        
        return json.loads(json_query)
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def search_opensearch(query):
    index_name = 'server_info'
    response = opensearch_client.search(
        index=index_name,
        body=query
    )
    return response['hits']['hits']

def natural_language_search(natural_language_query):
    opensearch_query = generate_opensearch_query(natural_language_query)
    
    print("\nGenerated OpenSearch Query: {}".format(opensearch_query))
    
    search_results = search_opensearch(opensearch_query)
    return search_results

# 메인 실행
if __name__ == "__main__":
    # natural_language_query = "Find all servers with more than 16GB of memory that are currently running"
    # opensearch_query = generate_opensearch_query(natural_language_query)

    # print(json.dumps(opensearch_query, indent=2))

    user_query = "Find all linux servers that are currently running"
    results = natural_language_search(user_query)
    
    print("\nSearch Results:")
    for result in results:
        print(json.dumps(result['_source'], indent=2))
    print(f"\nTotal results: {len(results)}")
