from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-normas"
EMBEDDING_MODEL = "text-embedding-3-small"

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

pinecone_client = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pinecone_client.Index(INDEX_NAME)

def get_relevant_documents(query):
        query_embedding_response = client.embeddings.create(
                input=query,
                model=EMBEDDING_MODEL
        )
        query_embedding = query_embedding_response.data[0].embedding
        relevant_documents = index.query(
                vector=query_embedding, 
                top_k=1, 
                include_metadata=True
        )
        return relevant_documents["matches"][0]["metadata"]["text"]

def process_query(query, n_results = 1):
        relevant_document = get_relevant_documents(query)
        query_with_context = f'####{query}####\nInformación: {relevant_document}'
        return query_with_context

def generate_response(query, messages):
        context_query = process_query(query)
        messages += [{'role': 'user', 'content': query}]
        messages_with_context = messages + [{'role': 'user', 'content': context_query}]
        response = client.chat.completions.create(
                messages=messages_with_context,
                model='gpt-3.5-turbo'
        ).choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}]
        return messages

