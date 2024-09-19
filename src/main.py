from chatting import persona_agent
import streamlit as st
import os

def chat_page():
    
    pdf_list = ["spiderman1.pdf", "spiderman2.pdf"]
    agent = persona_agent(pdf_list)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"].write(msg["content"]))
    
    
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            assistant_response = agent.receive_chat(prompt)
            
            message_placeholder.write(assistant_response)
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

def main():
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
    
    chat_page()

if __name__ == "__main__":
    main()