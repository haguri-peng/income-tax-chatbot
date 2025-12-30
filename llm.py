import os

from operator import itemgetter  # 딕셔너리에서 특정 키 값을 추출할 때 사용 (람다 대신 깔끔하게)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_xai import ChatXAI

from config import answer_examples


# ------------------ 싱글톤 캐싱 ------------------
_retriever = None
_llm = None
_rag_chain = None

# ------------------------------------------------------------
# 1. 세션별 대화 히스토리 저장소 (메모리 기반)
# ------------------------------------------------------------
# Streamlit은 페이지 리로드될 때마다 코드가 다시 실행되므로,
# 전역 변수 store에 세션별로 ChatMessageHistory 객체를 보관합니다.
# key = session_id (예: "default", "user123" 등)
store = {}


def get_session_history(session_id: str):
    """
    특정 세션 ID에 해당하는 대화 히스토리를 반환.
    없으면 새로 만들어서 store에 저장 후 반환.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ------------------------------------------------------------
# 2. Retriever (Pinecone 벡터 DB에서 문서 검색)
# ------------------------------------------------------------
def get_retriever():
    global _retriever
    if _retriever is None:
        print("[Initializer] Retriever 초기화 중...")

        # 임베딩 모델 로드
        embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

        index_name = "tax-markdown-index"
        pinecone_api_key = os.getenv("PINECONE_API_KEY")

        # Pinecone 클라이언트 초기화 (인덱스에 직접 연결)
        Pinecone(api_key=pinecone_api_key)

        # 기존 인덱스에서 VectorStore 객체 생성
        database = PineconeVectorStore.from_existing_index(
            embedding=embedding,
            index_name=index_name,
        )

        # k=2 → 가장 유사한 2개 문서를 가져오도록 설정
        _retriever = database.as_retriever(search_kwargs={"k": 2})

        print("[Initializer] Retriever 초기화 완료")
    return _retriever


# ------------------------------------------------------------
# 3. LLM (Grok 모델)
# ------------------------------------------------------------
def get_llm():
    global _llm
    if _llm is None:
        print("[Initializer] LLM 초기화 중...")
        _llm = ChatXAI(model="grok-4-1-fast-reasoning")
        print("[Initializer] LLM 초기화 완료")
    return _llm


# ------------------------------------------------------------
# 4. 질문 재작성 체인 (Dictionary Chain)
#    → "직장인" 같은 표현을 "거주자"로 바꿔서 검색 정확도 높임
# ------------------------------------------------------------
def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]  # 필요 시 더 추가 가능

    prompt = ChatPromptTemplate.from_template(
        """
        사용자의 질문을 보고, 우리의 사전을 참조해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 바꾸지 않아도 되며, 이런 경우에는 사용자의 질문을 그대로 반환해주세요.

        사전: {dictionary}
        질문: {question}

        답변은 변경된 질문만 반환하고, 추가 설명은 하지 마세요.
        """
    ).partial(dictionary=str(dictionary))

    # {"input": 원본 질문} → prompt → LLM → 문자열 파싱
    base_chain = {"question": itemgetter("input")} | prompt | get_llm() | StrOutputParser()

    # 디버깅용 print 함수 (원본 vs 재작성된 질문 확인)
    def print_and_return(rewritten: str, original: str):
        print("\n[Dictionary 적용 확인]")
        print(f"원본 질문: {original}")
        print(f"Rewrite된 질문: {rewritten}")
        print("-" * 50)
        return rewritten

    # RunnableParallel로 여러 값을 동시에 처리하고,
    # 마지막에 print_and_return를 거쳐 재작성된 질문만 반환
    return RunnableParallel({
        "rewritten_question": base_chain,
        "original_input": itemgetter("input"),
        "chat_history": itemgetter("chat_history")
    }) | RunnableLambda(lambda x: print_and_return(x["rewritten_question"], \
                                                   x["original_input"]) or x["rewritten_question"])


# ------------------------------------------------------------
# 5. 검색된 문서 포맷팅 + 디버깅 출력
# ------------------------------------------------------------
def format_docs_with_print(docs):
    print("\n[Retriever 확인]")
    print(f"검색된 문서 수: {len(docs)}")
    for i, doc in enumerate(docs, 1):
        print(f"\n문서 {i} (metadata: {doc.metadata})")
        print(f"내용 미리보기: {doc.page_content[:500]}...")  # 전체 보고 싶으면 [:500] 제거
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"\n최종 컨텍스트 길이: {len(context)} 문자")
    print("-" * 50)
    return context


# ------------------------------------------------------------
# 6. 전체 RAG 체인 구성
# ------------------------------------------------------------
def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        print("[Initializer] RAG Chain 초기화 중...")
        _rag_chain = create_rag_chain()
        print("[Initializer] RAG Chain 초기화 완료")
    return _rag_chain


def create_rag_chain():
    llm = get_llm()
    retriever = get_retriever()
    dictionary_chain = get_dictionary_chain()

    # Few-shot 예시 프롬프트 (config.py에 정의된 예시들)
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    # 시스템 프롬프트 (핵심 지시사항)
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원하며, 구체적으로 금액까지 언급해주세요"
        "\n\n"
        "{context}"
    )

    # 최종 QA 프롬프트 (시스템 + few-shot + 대화 히스토리 + 사용자 질문)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 전체 파이프라인
    rag_chain = (
        # 1) 원본 입력, 재작성 질문, 히스토리 병렬 처리
        RunnableParallel({
            "original_input": itemgetter("input"),
            "rewritten_question": dictionary_chain,
            "chat_history": itemgetter("chat_history")
        })
        # 2) 재작성된 질문으로 검색 → 컨텍스트 생성 (디버깅 print 포함)
        | RunnableParallel({
            "context": itemgetter("rewritten_question") \
                | retriever | RunnableLambda(format_docs_with_print),
            "input": itemgetter("original_input"),
            "chat_history": itemgetter("chat_history")
        })
        # 3) 프롬프트에 넣고 LLM 호출 → 문자열 출력
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ------------------------------------------------------------
# 7. AI 응답 생성 함수 (Streamlit에서 호출)
# ------------------------------------------------------------
def get_ai_response(user_question: str, session_id: str = "default"):
    """
    - 전체 RAG 체인을 매번 새로 생성 (함수 내부에서 호출)
    - 해당 세션의 히스토리를 가져와서 전달
    - stream()으로 실시간 토큰을 yield → Streamlit의 st.write_stream에서 사용
    - 전체 응답이 완성된 후 히스토리에 저장 (대화 기록 유지)
    """
    rag_chain = get_rag_chain()
    history = get_session_history(session_id)

    print(f"[User Input] 질문: {user_question}")
    print("=" * 70)

    # 스트리밍 응답 생성
    response_stream = rag_chain.stream({
        "input": user_question,
        "chat_history": history.messages  # List[BaseMessage] 형태로 전달
    })

    full_response = ""
    for chunk in response_stream:
        yield chunk  # Streamlit이 실시간으로 화면에 출력
        full_response += chunk

    print("\n" + "=" * 70)
    print("[AI Response] " + full_response)

    # 대화 히스토리에 저장 (다음 질문 시 컨텍스트로 사용)
    history.add_user_message(user_question)
    history.add_ai_message(full_response)
