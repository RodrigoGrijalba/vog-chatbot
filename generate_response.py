from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-normas"
EMBEDDING_MODEL = "text-embedding-3-small"

classification_prompt = """
Eres una trabajadora social enfocada en brindar apoyo en casos de violencia \
obstétrica. Tu tarea es determinar si el testimonio provisto por el \
usuario puede ser considerado un caso de violencia obstétrica. Para esto, \
se te proveerá de información sobre la normativa vigente para la práctica \
gineco-obstétrica. Utiliza únicamente esta información para determinar si \
el testimonio representa un caso de violencia. En caso se trate de un \
caso, utiliza la información provista para presentar las razones por qué.
En tu respuesta, mantén un tono amigable, cálido, y empático.
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

