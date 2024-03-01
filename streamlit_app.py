import openai
import streamlit as st
from dotenv import load_dotenv
import os
from html_template import *
from generate_response import classification_prompt, generate_response

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv('OPENAI_API_KEY')

def response_from_query():
    if st.session_state.prompt == "":
        return
    
    messages = st.session_state.history

    messages = generate_response(st.session_state.prompt, messages)
    st.session_state.history = messages

def main():
        
    if "history" not in st.session_state:
        st.session_state.history = [{'role': 'system', 'content': classification_prompt}]
    
    for message in st.session_state.history:
        if message["role"] == 'user':
            st.write(user_msg_container_html_template.replace("$MSG", message["content"]), unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.write(bot_msg_container_html_template.replace("$MSG", message["content"]), unsafe_allow_html=True)
    
    st.text_input(
        "Consulta al asistente virtual:", 
        key="prompt", 
        placeholder="ej.: Menciónenos qué ha sucedido durante la ateción obstétrica", 
        on_change=response_from_query
    )

if __name__ == "__main__":
    main()