import openai
import streamlit as st
import os
from html_template import *
from generate_response import classification_prompt, generate_response
from supabase import create_client, Client
import uuid

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv('OPENAI_API_KEY')
DATABASE_NAME = "vog-chatbot"

supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

def insert_data(uuid, message, table = DATABASE_NAME):
    data = {"uuid": uuid, "role": message["role"], "content": message["content"]}
    row_insert = supabase.table(table).insert(data)
    return row_insert

def session_id():
    return str(uuid.uuid4())

def clear_text_box():
    st.session_state.temp = st.session_state.prompt
    st.session_state.prompt = ""

def response_from_query():
    clear_text_box()
    if st.session_state.temp == "":
        return
    
    messages = st.session_state.history

    messages = generate_response(st.session_state.temp, messages)
    st.session_state.history = messages
    insert_data(st.session_state.session_id, messages[-2]).execute()
    insert_data(st.session_state.session_id, messages[-1]).execute()


def main():

    if "session_id" not in st.session_state:
        st.session_state.session_id = session_id()
        
    if "history" not in st.session_state:
        st.session_state.history = [{'role': 'system', 'content': classification_prompt}]
    
    if "temp" not in st.session_state:
        st.session_state.temp = ""
    
    for message in st.session_state.history:
        if message["role"] == 'user':
            st.write(user_msg_container_html_template.replace("$MSG", message["content"]), unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.write(bot_msg_container_html_template.replace("$MSG", message["content"]), unsafe_allow_html=True)
    
    st.text_area(
        "Hola, soy Illa. Encantada de conocerte. Estoy aquí para ayudarte", 
        value="",
        key="prompt", 
        placeholder="Cuéntame qué te sucedió durante la atención obstétrica o ginecológica", 
        on_change=response_from_query
    )

if __name__ == "__main__":
    main()