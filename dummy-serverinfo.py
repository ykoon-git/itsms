import json 
import time 
import random
from faker import Faker
from datetime import datetime, timedelta

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
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
index_name = 'server_info'

# 인덱스 생성 함수
def create_index_if_not_exists():
    index_mapping = {
        "mappings": {
            "properties": {
                "instance_name": {"type": "keyword"},
                "cpu": {"type": "integer"},
                "memory": {"type": "integer"},
                "disk": {"type": "integer"},
                "os": {"type": "text"},
                "purpose": {"type": "text"},
                "service_name": {"type": "text"},
                "ip_address": {"type": "ip"},
                "location": {"type": "text"},
                "department": {"type": "text"},
                "last_updated": {"type": "date"},
                "registration_date": {"type": "date"},
                "server_status": {"type": "text"},
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
    """
    1초 간격으로 인덱스 생성을 확인하고, 생성되면 루프를 종료합니다.
    
    :param index_name: 확인할 인덱스의 이름
    :param timeout: 최대 대기 시간 (초)
    :return: 인덱스 생성 성공 여부 (Boolean)
    """
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
    
# 서버 정보 생성 함수
def generate_server_info():
    registration_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 302))
    server_status = random.choice(['running', 'shutdown', 'stop'])
    company_name = random.choice(['SK Discovery', 'SK Chemical', 'SK GAS'])
    return {
        "instance_name": f"srv-{fake.random_int(min=1000, max=9999)}",
        "cpu": random.choice([2, 4, 8, 16, 32, 64]),
        "memory": random.choice([4, 8, 16, 32, 64, 128]),
        "disk": random.choice([100, 200, 500, 1000, 2000]),
        "os": random.choice(["Ubuntu 20.04", "CentOS 7", "Windows Server 2019", "Red Hat Enterprise Linux 8"]),
        "purpose": random.choice(["Web Server", "Database Server", "Application Server", "File Server", "Backup Server"]),
        "service_name": company_name,
        "ip_address": fake.ipv4(),
        "location": fake.city(),
        "department": random.choice(["IT", "Finance", "HR", "Marketing", "Sales", "R&D"]),
        "last_updated": fake.date_time_this_year().isoformat(),
        "registration_date": registration_date.isoformat(),
        "server_status": server_status
        # "vector_embedding": [random.random() for _ in range(768)]  # 임의의 벡터 생성
    }

# 더미 데이터 생성 및 인덱싱
def index_dummy_data(num_records):
    for _ in range(num_records):
        server_info = generate_server_info()
        embed_info = generate_embedding(server_info)
        server_info.update({"vector_embedding": embed_info})
        
        response = client.index(
            index=index_name,
            body=server_info,
            # refresh=True
        )
        print(f"Indexed document ID: {response['_id']}")

# 메인 실행
if __name__ == "__main__":
    create_index_if_not_exists()
    
    wait_for_index_creation (index_name)
    
    num_records = 100  # 생성할 레코드 수
    index_dummy_data(num_records)
    print(f"{num_records} dummy records have been indexed to OpenSearch Serverless.")
