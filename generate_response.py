from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-normas"
EMBEDDING_MODEL = "text-embedding-3-small"

classification_prompt = """
Eres una trabajadora social enfocada en brindar información en casos de \
violencia obstétrica. Tu tarea es determinar si el mensaje provisto por el \
usuario, puede ser considerado un caso de violencia obstétrica. Para esto, \
se te proveerá de información sobre la normativa vigente para la práctica \
gineco-obstétrica. Utiliza únicamente esta información para determinar si \
el mensaje representa un caso de violencia. En caso se trate de un \
caso, utiliza la información provista para presentar las razones por qué. \
Para cada texto de información provisto, se proveerá también el título del \
documento, el autor, el año, y el URL para acceder al documento. Al \
justificar tu respuesta, apóyate únicamente de la información y menciona el \
título, autor, año y URL del documento.

El mensaje del usuario estará delimitado por los siguientes caracteres: ####. \
Si el mensaje del usuario no está relacionado a la violencia obstétrica o \
ginecológica, responde de manera conversacional solamente al contenido \
demarcado por ####, sin tomar en cuenta la información adicional. Menciona \
que puedes brindar información sobre temas de violencia obstétrica y \
ginecológica.

En tu respuesta, mantén un tono amigable, cálido, y empático.
"""

CONTEXT_TEMPLATE = """
Información: {text}

Título: {title}
Autor: {author}
Año: {year}
URL: {url}
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
        return relevant_documents["matches"][0]["metadata"]

def process_query(query, n_results = 1):
        relevant_document = get_relevant_documents(query)
        context = CONTEXT_TEMPLATE.format(
                text=relevant_document["text"],
                title=relevant_document["title"],
                author=relevant_document["author"],
                year=relevant_document["year"],
                url=relevant_document["url"]
        )
        query_with_context = f'####{query}####\nInformación: {relevant_document}'
        return query_with_context

def generate_response(query, messages):
        context_query = process_query(query)
        messages += [{'role': 'user', 'content': query}]
        messages_with_context = messages + [{'role': 'user', 'content': context_query}]
        response = client.chat.completions.create(
                messages=messages_with_context,
                model='gpt-4-turbo-preview',
                stream=True
        )
        return messages, response

