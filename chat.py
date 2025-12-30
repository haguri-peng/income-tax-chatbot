import streamlit as st

from dotenv import load_dotenv
from llm import get_ai_response, get_rag_chain

load_dotenv()

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

# 앱 최초 실행 시 RAG 체인 미리 초기화 (캐싱 트리거)
with st.spinner("소득세 챗봇 초기화 중입니다... (최초 1회만 걸려요 🤖)"):
    get_rag_chain()  # 미리 호출해서 캐싱 트리거

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# ========== 대화 ==========
# 기존 대화 기록 유지
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전 메시지들 렌더링
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):  # "user" or "ai"
        st.markdown(message["content"], unsafe_allow_html=True)

# 사용자 입력
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    # 사용자 메시지 즉시 추가 및 표시
    st.session_state.message_list.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # AI 응답 버블 생성
    with st.chat_message("ai"):
        message_placeholder = st.empty()  # 실시간 업데이트용 placeholder
        full_response = ""

        # 스피너와 함께 스트리밍
        with st.spinner("답변을 생성하는 중입니다..."):
            for chunk in get_ai_response(user_question):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")  # 커서 효과

        # 스트리밍 완료 후 최종 텍스트 표시
        message_placeholder.markdown(full_response)

    # 스트리밍이 완전히 끝난 후에만 히스토리에 추가 → rerun 1회만 발생
    st.session_state.message_list.append({"role": "ai", "content": full_response})
