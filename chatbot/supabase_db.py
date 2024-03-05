import os, streamlit as st, time, uuid
from supabase import create_client, Client
from dotenv import find_dotenv, load_dotenv

from typing import Literal

# Environ
load_dotenv(find_dotenv())

url = os.environ.get("supabase_url")
key = os.environ.get('supabase_key')

table_name = 'vog-chatbot'


supabase: Client = create_client(url, key)


# functions
def insert_data(uuid: str, role:Literal['user', 'assistant'] = 'user', content: str = None, table = table_name):
    data = {"uuid": uuid, "role": role, 'content': content}
    row_insert = supabase.table(table).insert(data).execute()
    return row_insert


def collect_data_df(table = table_name):
    data = supabase.table(table).select('*').order("created_at", desc=True).execute()
    return data.data

def uuid_user():
    return str(uuid.uuid4())
################ ===


# Example
def response_from_query():

    user = 'user_uuid'

    if user not in st.session_state:
        st.session_state[user] = uuid_user()
    
    question = st.text_input("Ingrese la pregunta")
    if st.button('generar respuesta'):
        time.sleep(1)
        response = f"La pregunte fue: {question}"
        # st.write(response)

        insert_data(st.session_state[user], 'user', question)
        insert_data(st.session_state[user], 'assistant', response)


if __name__ == '__main__':
    response_from_query()




