import os   #운영 체제와 상호 작용하기 위한 수십 가지 함수를 제공하는 모듈
from fastapi import FastAPI #빠르고 사용하기 쉬운 ASGI 웹 프레임워크로 API 생성에 사용
from fastapi.responses import RedirectResponse  #응답 클래스 중 하나로, 특정 URL로 리다이렉션 수행
from fastapi.middleware.cors import CORSMiddleware  #CORS는 Cross-Origin Resource Sharing의 약자로, 다른 출처 간의 자원 공유 시 사용 
from typing import List, Union  #Python의 typing 모듈은 코드에 타입 @힌트를 추가하기 위한 용도 
from langserve.pydantic_v1 import BaseModel, Field  #데이터 검증을 위한 클래스입니다.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  #메시지(사람, AI, 시스템)를 다루는 클래스를 임포트
from langserve import add_routes    #FastAPI 애플리케이션에 라우트를 추가하는 함수
from langchain_community.chat_models import ChatOllama  #Ollama 채팅 모델을 핸들링하는 클래스
from langchain_core.output_parsers import StrOutputParser   #출력을 파싱하여 문자열 형태로 만드는 클래스
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  #langchain_core.prompts로부터 프롬프트를 관리하는 클래스를 가져옮

from dotenv import load_dotenv
# .env 파일 내의 변수들을 로드합니다.
load_dotenv()

######## ----------------------------------------------------------------
# 클라이언트의 요청에 따라 특정 작업(프롬프트 생성, 챗 모델을 통한 대화 생성, 문장 번역 등)을 수행하고 그 결과를 반환하는 역할을 합니다.
#### -- REST API 서버의 구현에 해당하는 코드로 볼 수 있으며, 이를 통해 클라이언트는 서버에 요청을 보내 이런 기능들을 사용할 수 있게 됩니다. 

# LangChain의 챗모델 Ollama를 로드합니다.
# 여기서 모델을 지정할 때는 원하는 모델의 이름을 문자열로 입력합니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

# 프롬프트를 생성합니다. 여기서는 사용자가 주어진 주제에 대해 설명하도록 지시하는 프롬프트를 만듭니다.
prompt_prompt = ChatPromptTemplate.from_template("{topic} 에 대하여 간략히 설명해 줘.")

# 프롬프트, 챗모델, 출력 parser를 체인으로 연결합니다. 이 체인은 사용자의 입력을 받아 챗모델을 거쳐 파싱된 출력을 만듭니다.
chain = prompt_prompt | llm | StrOutputParser()

# ChatPromptTemplate의 from_messages 메소드를 사용하여 채팅 프롬프트를 생성합니다.
chat_prompt = ChatPromptTemplate.from_messages(
    [
        # 이 배열의 첫 번째 요소는 시스템 메시지입니다. 이것은 AI에게 명령을 내립니다.
        # 이 경우에는, 내 이름은 '테디'이고 도움을 주는 AI Assistant임을 명시하고 있습니다. 이 AI는 반드시 한국어로 대답해야 합니다.
        (
            "system",
            "You are a helpful AI Assistant. Your name is '테디'. You must answer in Korean.",
        ),
        # 이 프롬프트는 사용자의 메시지를 저장하는 Placeholder를 포함하고 있습니다.
        # Placeholder는 템플릿에서 변하는 부분을 나타내는데 사용됩니다.
        # 'variable_name' 속성의 값으로 'messages'를 설정하여, 채팅 대화에서 사용자의 메시지를 포함하도록 합니다.
        MessagesPlaceholder(variable_name="messages")
    ]
)

# 'chat_prompt', 'llm', 그리고 'StrOutputParser'를 '|' 연산자를 이용하여 체인으로 연결합니다.
# 'chat_prompt'는 사용자의 입력을 받아 프롬프트를 생성하는 역할을 합니다.
# 'llm'은 생성된 프롬프트를 이용해 모델에서 대화를 생성하는 역할을 합니다.
# 'StrOutputParser'는 생성된 대화 결과를 파싱하여 텍스트 형태로 바꾸는 역할을 합니다.
# 이렇게 체인을 형성함으로써 사용자의 입력을 받아 대화를 생성하고, 그 결과를 텍스트로 반환하는 전체 과정을 한 라인에 구현하였습니다.
chat_chain = chat_prompt | llm | StrOutputParser()

# 번역 프롬프트를 생성합니다.
# 여기서는 주어진 문장들을 한국어로 번역하라는 지시를 가지는 프롬프트를 만듭니다.
translator_prompt = ChatPromptTemplate.from_template(
    "Translate following sentences into Korean:\n{input}"
)

# 프롬프트, 챗모델, 출력 parser를 체인으로 연결합니다.
# 이 체인은 사용자의 입력을 받아 프롬프트에 따라 챗모델을 거쳐 파싱된 출력을 만들어내는 역할을 합니다.
# 여기서는 주어진 문장을 한국어로 번역하는 작업을 수행합니다.
EN_TO_KO_chain = translator_prompt | llm | StrOutputParser()




######## ----------------------------------------------------------------
# REST API는 서버 파트에 해당합니다. REST API는 서버에서 데이터나 기능을 제공하는 엔드포인트를 의미하며, 클라이언트는 이러한 엔드포인트에 HTTP 요청을 보내어 필요한 데이터를 얻거나, 서버 상의 데이터를 변경/삭제하거나, 새로운 데이터를 생성하는 등의 작업을 수행할 수 있습니다. 따라서 REST API는 서버에서 제공되는 서비스로 볼 수 있고, 클라이언트는 이러한 서비스를 사용하는 방향으로 개발됩니다.



#### -- 이 코드는 FastAPI 앱을 초기화하고 CORS(Cross-Origin Resource Sharing)를 설정하는 부분입니다. 이 부분도 웹 API를 제공하는 흐름 중 한 부분이므로 REST API 설정에 해당합니다. FastAPI인스턴스를 만들고 CORS 미들웨어를 추가함으로써, 이 API가 다른 도메인에서 접근하고 사용할 수 있게 됩니다. 이것이 없으면 보안 문제로 인해 API는 동일 도메인에서만 작동하게 설정됩니다. CORS 설정은 웹 서버가 다른 도메인, 포트, 스키마에서 동작하는 웹 페이지로부터 들어오는 요청을 어떻게 처리할지를 결정합니다.

# FastAPI는 Python으로 작성된 웹 프레임워크이며 API 생성에 사용됩니다.
app = FastAPI()

# CORS(Cross-Origin Resource Sharing) 미들웨어를 추가합니다.
# 이는 서로 다른 출처 간에 자원을 공유할 수 있게 해주는 기능입니다.
# 여기서 "*"는 모든 출처를 허용한다는 의미입니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

#### -- 주어진 FastAPI 애플리케이션을 통해 라우팅하는 RESTful API 서버를 설정하고 실행하는 부분입니다. 
# 다양한 경로("/prompt", "/chat", "/translate", "/llm")를 설정하여 
# 해당 경로로 REQUEST 요청이 들어왔을 때, 적절한 방식으로 처리하고 응답하는 역할을 합니다. 
# 마지막 부분에서는, 실행 시 uvicorn을 이용하여 이 FastAPI 애플리케이션(서버)을 0.0.0.0 주소의 8000 포트에서 동작하도록 합니다.

# 루트 URL("/")에 GET 요청이 왔을 경우 "/prompt/playground"로 리다이렉트 시킵니다.
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/prompt/playground")

# 기존의 체인(chain)을 "/prompt"라는 경로로 라우팅합니다.
# 이는 "/prompt" URL을 통해 체인의 기능에 접근할 수 있게 해줍니다.
add_routes(app, chain, path="/prompt")

# 채팅 엔드포인트를 위한 입력 타입을 정의합니다.
# 이는 대화를 구성하는 메시지 목록이며, 사람의 메시지, AI의 메시지, 시스템 메시지를 포함할 수 있습니다.
class InputChat(BaseModel):
    """Input for the chat endpoint."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )

# 채팅 체인(chat_chain)을 "/chat"라는 경로로 라우팅합니다.
# 이는 "/chat" URL을 통해 채팅 체인의 기능에 접근할 수 있게 해줍니다.
# 추가로, 피드백 엔드포인트, 공개 트레이스 링크 엔드포인트를 활성화하며, 
# 이 엔드포인트의 유형을 "chat"으로 설정합니다.    
add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

# 번역 체인(EN_TO_KO_chain)을 "/translate"라는 경로로 라우팅합니다.
# 이는 "/translate" URL을 통해 채팅 체인의 기능에 접근할 수 있게 해줍니다.
add_routes(app, EN_TO_KO_chain, path="/translate")

# ChatOllama 모델(llm)을 "/llm"라는 경로로 라우팅합니다.
# 이는 "/llm" URL을 통해 ChatOllama 모델에 직접 접근할 수 있게 해줍니다.
add_routes(app, llm, path="/llm")

# 이 스크립트가 메인 프로그램으로 실행되었을 때 아래의 코드를 실행합니다.
if __name__ == "__main__":
    
    # uvicorn은 ASGI 서버입니다. FastAPI 애플리케이션을 호스팅하기 위해 사용됩니다.
    # 여기서는 0.0.0.0 주소의 8000 포트에서 애플리케이션을 실행합니다.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)