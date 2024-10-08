from chatting import persona_agent
import streamlit as st
import os

from util import CHAT_ICON_LIST

def display_chat_message(profile_image, message, role):
    st.markdown(
        f'<div style="display: flex; align-items: center;">'
        f'<img src="{profile_image}" style="border-radius:50%; width: 30px; height: 30px; margin-right: 10px;">'
        f'{message}'
        f'</div><br>',
        unsafe_allow_html=True
    )

def chat_page(agent, char):
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
            "profile_image": CHAT_ICON_LIST[char]
        })
        display_chat_message(CHAT_ICON_LIST[char], assistant_response, "assistant")

def main():
    st.title("ChaCha ver2 - 캐릭터와 대화하기")

    st.sidebar.title("캐릭터 선택")
    selected_char = st.sidebar.selectbox(
        "대화할 캐릭터를 선택하세요",
        ["스파이더맨(피터 파커)",
        "전우치",
        "신짱구"]
    )

    if selected_char == "스파이더맨(피터 파커)":
        pdf_list = ["data/spiderman1.pdf", "data/spiderman2.pdf"]
        char = "pp"
    elif selected_char == "전우치":
        pdf_list = ["data/jwc.pdf"]
        char = "jwc"
    elif selected_char == "신짱구":
        pdf_list = ["data/szg1.pdf", "data/szg2.pdf"]
        char = "szg"

    agent = persona_agent(pdf_list, char)

    os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
    chat_page(agent, char)

if __name__ == "__main__":
    main()