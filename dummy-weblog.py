import json 
import time 
import random
from faker import Faker
from datetime import datetime, timedelta

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# AWS 인증 설정
region = 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

# OpenSearch Serverless 연결 설정
host = 'o0hj5d4vh1k6bxab969l.us-west-2.aoss.amazonaws.com' # vector
port = 443

# OpenSearch 클라이언트 생성
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Faker 인스턴스 생성
fake = Faker()

# 인덱스 이름 설정
index_name = 'weblog_info'

# 인덱스 생성 함수
def create_index_if_not_exists():
    index_mapping = {
        "mappings": {
            "properties": {
                "timestamp": {"type": "date"},
                "ip_address": {"type": "ip"},
                "method": {"type": "keyword"},
                "url": {"type": "text"},
                "status_code": {"type": "integer"},
                "user_agent": {"type": "text"},
                "referrer": {"type": "text"},
                "response_time": {"type": "float"},
                "bytes_sent": {"type": "long"},
                "full_text":{"type": "text"},
                "vector_embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2"
                    }
                }
            }
        }
    }
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_mapping)
        print(f"Index '{index_name}' created with required mappings.")
    else:
        print(f"Index '{index_name}' already exists.")

def check_index_exists(index_name):
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        print(f"Error checking index: {str(e)}")
        return False

def wait_for_index_creation(index_name, timeout=300):
    start_time = time.time()
    
    while True:
        if time.time() - start_time > timeout:
            print(f"Timeout: Index '{index_name}' was not created within {timeout} seconds.")
            return False
        
        if check_index_exists(index_name):
            print(f"Index '{index_name}' has been successfully created.")
            time.sleep(5) # 추가 시간 대기
            return True
        
        print(f"Waiting for index '{index_name}' to be created...")
        time.sleep(1)

# Server Info Embedding
def generate_embedding(json_obj, dimensions=1024, normalize=True):
    """
    Amazon Titan Text Embeddings V2를 사용하여 문자열을 임베딩합니다.
    
    :param text: 임베딩할 문자열
    :param dimensions: 임베딩 벡터의 차원 (256, 512, 또는 1024)
    :param normalize: 임베딩 벡터를 정규화할지 여부
    :return: 임베딩 벡터
    """
    # Bedrock 클라이언트 생성
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2'  # 사용하려는 AWS 리전으로 변경하세요
    )
    
    text = json.dumps(json_obj)
    
    # 요청 본문 생성
    body = json.dumps({
        "inputText": text,
        "dimensions": dimensions,
        "normalize": normalize
    })
    
    try:
        # 모델 호출
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="amazon.titan-embed-text-v2:0",
            accept="application/json",
            contentType="application/json"
        )
        
        # 응답 처리
        response_body = json.loads(response.get('body').read())
        embedding = response_body['embedding']
        
        return embedding
    
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None
    
# 웹 서버 로그 생성 함수
def generate_web_log():
    timestamp = datetime(2024, 10, 1) + timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    return {
        "timestamp": timestamp.isoformat(),
        "ip_address": fake.ipv4(),
        "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
        "url": fake.uri_path(),
        "status_code": random.choice([200, 201, 204, 400, 401, 403, 404, 500]),
        "user_agent": fake.user_agent(),
        "referrer": fake.uri(),
        "response_time": round(random.uniform(0.1, 2.0), 3),
        "bytes_sent": random.randint(500, 5000),
    }

# 더미 데이터 생성 및 인덱싱
def index_dummy_data(num_records):
    for _ in range(num_records):
        web_log = generate_web_log()
        full_text = json.dumps(web_log) # 전체 json 텍스트 저장용
        embed_info = generate_embedding(web_log)
        web_log.update({"full_text": full_text})
        web_log.update({"vector_embedding": embed_info})
        
        response = client.index(
            index=index_name,
            body=web_log,
            # refresh=True
        )
        print(f"Indexed document ID: {response['_id']}")

# 메인 실행
if __name__ == "__main__":
    create_index_if_not_exists()
    
    wait_for_index_creation(index_name)
    
    num_records = 500  # 생성할 레코드 수
    index_dummy_data(num_records)
    print(f"{num_records} dummy web log records have been indexed to OpenSearch Serverless.")
