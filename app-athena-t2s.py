import os
import streamlit as st
from langchain_community.llms import Bedrock
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate

from langchain_aws import ChatBedrock
from langchain_community.agent_toolkits import create_sql_agent

from sqlalchemy import create_engine

# 사용할 LLM 모델을 선택하고 파라미터값을 설정합니다.
@st.cache_resource
def get_llm():
    model_kwargs =  { #Anthropic 모델
        "max_tokens": 8000,
        "temperature": 0, 
        "top_k": 250, 
        "top_p": 0.5, 
        "stop_sequences": ["\n\nHuman:"] 
    }
    
    # Bedrock 의 Anthropic Claude 3.5 Sonnet 을 사용합니다. 
    llm = ChatBedrock(
        credentials_profile_name='default',
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs=model_kwargs)
    
    return llm

@st.cache_resource
def get_athena_agent():
    llm = get_llm()

    # Athena 를 통해서 DataLake 에 연결합니다.
    conn_str = "awsathena+rest://athena.us-west-2.amazonaws.com:443/"\
               "itsms?s3_staging_dir=s3://athena-federation-20240224/athenaresults/"

    engine = create_engine(conn_str.format(
        region_name="us-west-2",
        schema_name="itsms",
        s3_staging_dir="s3://athena-federation-20240224/athenaresults/"))
        
    athena_db_connection = SQLDatabase(engine)
    athena_agent_executor = create_sql_agent(llm, db=athena_db_connection, verbose=True)
    
    return athena_agent_executor

# Streamlit 앱 설정
st.title("Athena DataLake(Server Info) Chatbot")

# 사이드바에 사용 설명 추가
st.sidebar.header("How to use")
st.sidebar.write("1. 아래의 텍스트박스에 질문을 입력합니다.(ex. windows 서버의 수량은?)")
st.sidebar.write("2. Chatbot 이 Database 의 Schema 등을 확인하고 질문을 SQL query 로 변환합니다.")
st.sidebar.write("3. 만들어진 쿼리를 Athena 쿼리 엔진을 통해 실행합니다.")

# 사용자 입력 받기
user_input = st.text_input("서버정보를 조회하기 위한 질문을 입력하세요:", "")

if user_input:
    athena_agent_executor = get_athena_agent()
    
    with st.spinner('질문을 처리중입니다 (처리중인 내용은 콘솔에서 확인할 수 있습니다)...'):
        # 질문을 전달하고 결과 받기
        response = athena_agent_executor.invoke(user_input)
        
        # 결과 표시
        st.subheader("Response:")
        st.write(response)
        
        # SQL 쿼리 표시 (만약 반환된 response에 SQL 쿼리가 포함되어 있다면)
        if 'sql_query' in response:
            st.subheader("Generated SQL Query:")
            st.code(response['sql_query'], language='sql')

# 추가 정보 표시
st.markdown("---")
st.write("이 챗봇은 Amazon Bedrock의 Claude 3.5 Sonnet 모델을 사용하여 사용자의 질문을 해석하고 Athena 데이터베이스에 대한 SQL 쿼리를 생성합니다.")
