import streamlit as st
import boto3
from datetime import datetime
import json

class BedrockPromptManager:
    def __init__(self, region: str = "us-east-1"):
        self.bedrock_client = boto3.client(
            service_name="bedrock-agent",
            region_name=region
        )
        self.runtime_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
    
    def create_prompt(self, name: str, content: str, description: str, model_id: str):
        try:
            response = self.bedrock_client.create_prompt(
                name=name,
                description=description,
                variants=[{
                    "name": "default",
                    "modelId": model_id,
                    "templateType": "TEXT",
                    "inferenceConfiguration": {
                        "text": {
                            "temperature": st.session_state.temperature,
                            "topP": st.session_state.top_p,
                        }
                    },
                    "templateConfiguration": {
                        "text": {
                            "text": content
                        }
                    }
                }]
            )
            return response
        except Exception as e:
            return {"error": str(e)}

    def list_prompts(self):
        try:
            response = self.bedrock_client.list_prompts()
            return response.get("promptSummaries", [])
        except Exception as e:
            return []

    def get_prompt(self, prompt_id: str, version: str = None):
        try:
            params = {"promptIdentifier": prompt_id}
            if version:
                params["promptVersion"] = version
            response = self.bedrock_client.get_prompt(**params)
            return response
        except Exception as e:
            return {"error": str(e)}

    def invoke_prompt(self, prompt_id: str, variables: dict, version: str = None):
        try:
            prompt_info = self.get_prompt(prompt_id, version)
            if "error" in prompt_info:
                return prompt_info

            variant = prompt_info["variants"][0]
            model_id = variant["modelId"]
            template = variant["templateConfiguration"]["text"]["text"]
            
            # 변수 치환
            for key, value in variables.items():
                template = template.replace(f"{{{{{key}}}}}", value)

            response = self.runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "prompt": template,
                    "temperature": variant["inferenceConfiguration"]["text"]["temperature"],
                    "topP": variant["inferenceConfiguration"]["text"].get("topP", 0.9)
                })
            )
            
            return json.loads(response['body'].read())
        except Exception as e:
            return {"error": str(e)}

def get_prompt_and_inject_variables(self, prompt_id: str, variables: dict) -> str:
    """프롬프트 템플릿과 변수를 병합하여 최종 프롬프트를 생성"""
    try:
        # 프롬프트 정보 가져오기
        prompt_info = self.get_prompt(prompt_id)
        if "error" in prompt_info:
            raise ValueError(f"프롬프트 정보 조회 실패: {prompt_info['error']}")

        # 템플릿 텍스트 추출
        template = prompt_info["variants"][0]["templateConfiguration"]["text"]["text"]

        # 변수 치환
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in template:
                template = template.replace(placeholder, str(value))

        return template

    except Exception as e:
        raise Exception(f"프롬프트 변수 치환 중 오류 발생: {str(e)}")

def execute_prompt_with_llm(self, prompt_id: str, variables: dict) -> str:
    """LLM에 프롬프트를 실행하고 결과를 반환"""
    try:
        # 프롬프트 준비
        prepared_prompt = self.get_prompt_and_inject_variables(prompt_id, variables)
        
        # 프롬프트 정보 가져오기
        prompt_info = self.get_prompt(prompt_id)
        model_id = prompt_info["variants"][0]["modelId"]
        
        # 추론 설정 가져오기
        inference_config = prompt_info["variants"][0]["inferenceConfiguration"]["text"]
        temperature = inference_config.get("temperature", 0.7)
        top_p = inference_config.get("topP", 0.9)

        # Bedrock 런타임 호출
        response = self.runtime_client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "prompt": prepared_prompt,
                "temperature": temperature,
                "topP": top_p,
                "maxTokens": 1000
            })
        )

        # 응답 처리
        response_body = json.loads(response['body'].read())
        
        # 모델별 응답 형식 처리
        if model_id.startswith('anthropic.'):
            return response_body.get('completion', '')
        elif model_id.startswith('amazon.'):
            return response_body.get('results', [{}])[0].get('outputText', '')
        else:
            return response_body.get('generated_text', '')

    except Exception as e:
        raise Exception(f"LLM 실행 중 오류 발생: {str(e)}")

def get_available_models(self) -> list:
    """사용 가능한 모델 목록 반환"""
    try:
        response = self.bedrock_client.list_foundation_models()
        return [
            model['modelId'] for model in response['modelSummaries']
            if model['modelLifecycle']['status'] == 'ACTIVE'
        ]
    except Exception as e:
        print(f"모델 목록 조회 실패: {str(e)}")
        return [
            'anthropic.claude-v2',
            'anthropic.claude-instant-v1',
            'amazon.titan-text-express-v1'
        ]


def main():
    st.set_page_config(page_title="Bedrock Prompt Management", layout="wide")
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.9

    st.title("🤖 Bedrock Prompt Management")
    
    tabs = st.tabs(["프롬프트 생성", "프롬프트 실행"])
    
    prompt_manager = BedrockPromptManager()

    # 프롬프트 생성 탭
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("새 프롬프트 생성")
            
            name = st.text_input("프롬프트 이름")
            description = st.text_input("설명")
            content = st.text_area(
                "프롬프트 내용",
                height=200,
                help="변수는 {{variable_name}} 형식으로 입력하세요."
            )
            
            model_id = st.selectbox(
                "모델 선택",
                ["amazon.titan-text-express-v1", "anthropic.claude-v2", "meta.llama2-70b"]
            )
            
            col_temp, col_top_p = st.columns(2)
            with col_temp:
                st.session_state.temperature = st.slider(
                    "Temperature", 0.0, 1.0, st.session_state.temperature
                )
            with col_top_p:
                st.session_state.top_p = st.slider(
                    "Top P", 0.0, 1.0, st.session_state.top_p
                )
            
            if st.button("프롬프트 저장", type="primary"):
                if name and content:
                    response = prompt_manager.create_prompt(
                        name=name,
                        content=content,
                        description=description,
                        model_id=model_id
                    )
                    
                    if "error" in response:
                        st.error(f"저장 실패: {response['error']}")
                    else:
                        st.success(f"프롬프트 '{name}' 저장 완료!")
                else:
                    st.warning("프롬프트 이름과 내용을 입력해주세요.")

        with col2:
            st.subheader("저장된 프롬프트")
            prompts = prompt_manager.list_prompts()
            
            if prompts:
                for prompt in prompts:
                    with st.expander(f"📝 {prompt['name']}"):
                        st.write(f"**설명:** {prompt.get('description', '설명 없음')}")
                        st.write(f"**ID:** {prompt.get('id', '정보 없음')}")
                        st.write(f"**생성일:** {prompt.get('createdAt', '')}")
                        st.write(f"**모델:** {prompt.get('modelId', '정보 없음')}")
            else:
                st.info("저장된 프롬프트가 없습니다.")

    # 프롬프트 실행 탭
    with tabs[1]:
        st.subheader("프롬프트 확인")
        
        prompts = prompt_manager.list_prompts()
        if prompts:
            selected_prompt = st.selectbox(
                "실행할 프롬프트 선택",
                options=[(p['id'], p['name']) for p in prompts],
                format_func=lambda x: x[1],
                key="prompt_selector"
            )
            
            if selected_prompt:
                prompt_id = selected_prompt[0]
                prompt_info = prompt_manager.get_prompt(prompt_id)
                
                if "error" not in prompt_info:
                    st.write("**프롬프트 정보:**")
                    st.json(prompt_info)
                    
                    # 변수 입력 필드 생성
                    st.subheader("변수 확인")
                    variables = {}
                    template = prompt_info["variants"][0]["templateConfiguration"]["text"]["text"]
                    import re
                    var_names = re.findall(r'\{\{(\w+)\}\}', template)
                    
                    import uuid
                    for idx, var in enumerate(var_names):
                        unique_key = f"var_{var}_{prompt_id}_{idx}_{uuid.uuid4().hex[:8]}"
                        variables[var] = st.text_input(
                            f"변수 {var}",
                            key=unique_key
                        )
        else:
            st.info("저장된 프롬프트가 없습니다.")


if __name__ == "__main__":
    main()
