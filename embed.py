import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import json

# AWS 설정
region = 'us-west-2'  # 사용 중인 리전으로 변경하세요
service = 'aoss'
credentials = boto3.Session().get_credentials()

awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

# OpenSearch Serverless 설정
host = 'o0hj5d4vh1k6bxab969l.us-west-2.aoss.amazonaws.com'  # 실제 엔드포인트로 변경하세요
index_name = 'itsmindex'




# OpenSearch 클라이언트 설정
client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

# Index 확인
response = client.indices.get_mapping(index="itsmindex")
print(response)

# #Index 생성
# index_settings = {
#     "settings": {
#         "index": {
#             "knn": True
#         }
#     },
#     "mappings": {
#         "properties": {
#             "embedding": {
#                 "type": "knn_vector",
#                 "dimension": 1024,  # 벡터의 차원을 지정하세요
#                 "method": {
#                     "name": "hnsw",
#                     "space_type": "l2",
#                     "engine": "nmslib"
#                 }
#             }
#         }
#     }
# }

# client.indices.create(index="itsmindex", body=index_settings)


# Bedrock 클라이언트 설정
bedrock = boto3.client(service_name='bedrock-runtime', region_name=region)

# 텍스트를 임베딩으로 변환하는 함수
def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

# 문서를 OpenSearch에 인덱싱하는 함수
def index_document(text, metadata=None):
    embedding = get_embedding(text)
    doc = {
        'text': text,
        'embedding': embedding
    }
    if metadata:
        doc.update(metadata)
    
    response = client.index(
        index = index_name,
        body = doc,
        # refresh = True
    )
    return response

# 샘플 사용
sample_text = "OpenSearch Serverless는 관리형 검색 및 분석 서비스입니다."
metadata = {"source": "AWS 블로그", "author": "AWS 팀"}

result = index_document(sample_text, metadata)
print(f"문서 인덱싱 결과: {result}")

# 벡터 검색 쿼리 예시
query_text = "OpenSearch Serverless의 특징은 무엇인가요?"
query_vector = get_embedding(query_text)

search_query = {
    "size": 5,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_vector,
                "k": 5
            }
        }
    }
}

search_result = client.search(
    body = search_query,
    index = index_name
)

print("검색 결과:")
for hit in search_result['hits']['hits']:
    print(f"Score: {hit['_score']}, Text: {hit['_source']['text']}")


result = client.count(index=index_name)
total_count = result['count']
print(f"총 문서 수: {total_count}")
