from openai import OpenAI
import streamlit as st
from chromadb import PersistentClient
from create_chroma import openai_embedding

classification_prompt = """
Eres un gineco obstetra calificado que cuenta con experiencia \
en situaciones de violencia ginecológica obstétrica
Se te proveerá información sobre la normativa de obsetétrica
Estudia la información \
provista y detecta si la el mensaje del usuario, delimitado por ####, representa alguna \
vulneración a la norma o que se evidencia alguna situación que no esté \
permitida como una práctica, procedimiento, maniobra o trato adecuado.
"""

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

chroma_client = PersistentClient('chroma')
collection = chroma_client.get_collection(
    name='vo-normas',
    embedding_function=openai_embedding
)


def process_query(query, n_results = 1):
    relevant_document = collection.query(
        query_texts=[query],
        n_results=n_results
    )['documents'][0][0]
    query_with_context = f'####{query}####\nInformación: {relevant_document}'
    return query_with_context

def generate_response(query, messages):
    context_query = process_query(query)
    classifications = []
    messages += [{'role': 'user', 'content': query}]
    messages_with_context = messages + [{'role': 'user', 'content': context_query}]
    response = client.chat.completions.create(
        messages=messages_with_context,
        model='gpt-3.5-turbo'
    ).choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    return messages

