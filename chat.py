import os
import streamlit as st

from dotenv import load_dotenv
from llm import get_ai_response, get_rag_chain

# .env 파일 로드
load_dotenv()

# 환경변수 확인
if not (os.getenv("XAI_API_KEY") and os.getenv("PINECONE_API_KEY")):
    st.error("API 키가 설정되지 않았습니다. [.env] 파일을 확인하세요.")
    st.stop()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

# 앱 최초 실행 시 RAG 체인 미리 초기화 (캐싱 트리거)
with st.spinner("소득세 챗봇 초기화 중입니다... (최초 1회만 🤖)"):
    try:
        get_rag_chain()  # 미리 호출해서 캐싱 트리거
        # st.session_state.rag_chain_initialized = True
    except Exception as e:
        st.error(f"챗봇 초기화 중 오류가 발생했습니다: {str(e)}. 다시 시도해주세요.")
        st.stop()  # 앱 중단 방지

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# # 세션 상태로 초기화 여부 관리
# if 'rag_chain_initialized' not in st.session_state:
#     st.session_state.rag_chain_initialized = False

# ========== 사이드바 ==========
with st.sidebar:
    if st.button("대화 리셋"):
        st.session_state.message_list = []
        # st.session_state.rag_chain_initialized = False  # 필요 시 재초기화
        st.rerun()

# ========== 대화 ==========
# 기존 대화 기록 유지
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전 메시지들 렌더링
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):  # "user" or "ai"
        # st.markdown(message["content"], unsafe_allow_html=True)
        st.markdown(message["content"])

# 메시지 제한 (Max: 50개)
MAX_MESSAGES = 50
if len(st.session_state.message_list) > MAX_MESSAGES:
    st.session_state.message_list = st.session_state.message_list[-MAX_MESSAGES:]

# 사용자 입력
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    # # 첫 질문일 때만 초기화
    # if not st.session_state.rag_chain_initialized:
    #     try:
    #         with st.spinner("소득세 챗봇 초기화 중입니다... (최초 1회만 걸려요 🤖)"):
    #             get_rag_chain()
    #             st.session_state.rag_chain_initialized = True
    #     except Exception as e:
    #         st.error(f"챗봇 초기화 중 오류가 발생했습니다: {str(e)}. 다시 시도해주세요.")
    #         st.stop()  # 앱 중단 방지

    # 사용자 메시지 즉시 추가 및 표시
    st.session_state.message_list.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # AI 응답 스트리밍
    with st.chat_message("ai"):
        message_placeholder = st.empty()  # 실시간 업데이트용 placeholder
        full_response = ""

        # 스피너와 함께 스트리밍
        try:
            with st.spinner("답변을 생성하는 중입니다..."):
                for chunk in get_ai_response(user_question):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")  # 커서 효과

            # 스트리밍 완료 후 최종 텍스트 표시
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}. 질문을 다시 입력해주세요.")

    # 스트리밍이 완전히 끝난 후에만 히스토리에 추가 → rerun 1회만 발생
    st.session_state.message_list.append({"role": "ai", "content": full_response})
