from chatting import persona_agent
import streamlit as st
import os

from .util import CHAT_ICON_LIST

def display_chat_message(profile_image, message, role):
    st.markdown(
        f'<div style="display: flex; align-items: center;">'
        f'<img src="{profile_image}" style="border-radius:50%; width: 30px; height: 30px; margin-right: 10px;">'
        f'{message}'
        f'</div><br>',
        unsafe_allow_html=True
    )

def chat_page():
    
    pdf_list = ["data/spiderman1.pdf", "data/spiderman2.pdf"]
    agent = persona_agent(pdf_list)
    
    st.title("ChaCha ver2 - 캐릭터와 대화하기")

    # 메세지 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 메세지 출력
    for msg in st.session_state.messages:
        display_chat_message(msg["profile_image"], msg["content"], msg["role"])
    
    # 새 인풋에 대한 출력 생성
    if prompt := st.chat_input("메세지를 입력하세요..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "profile_image": CHAT_ICON_LIST["user"]
        })

        display_chat_message(CHAT_ICON_LIST["user"], prompt, "user")

        #assistant_response = agent.receive_chat(prompt)

        assistant_response = agent.receive_chat(prompt)
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "profile_image": CHAT_ICON_LIST["peter parker"]
        })
        display_chat_message(CHAT_ICON_LIST["peter parker"], assistant_response, "assistant")

def main():
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
    chat_page()

if __name__ == "__main__":
    main()