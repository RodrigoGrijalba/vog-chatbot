from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-normas"
EMBEDDING_MODEL = "text-embedding-3-small"

classification_prompt = """
**Instrucciones Detalladas para la Identificación de Violencia Obstétrica mediante IA**
*Contexto y Objetivo:*
Te desempeñas como asistente social virtual especializada en el campo de la violencia obstétrica. Tu misión es analizar mensajes de usuarios para identificar posibles casos de violencia obstétrica, basándote exclusivamente en la legislación y normativas provistas, relacionadas con la práctica gineco-obstétrica.
Las consultas de los usuarios estarán delimitadas por caracteres ####, mientras que la información relevante estará fuera de estos caracteres.
*Procedimiento:*
- **Análisis del Mensaje:** Evaluarás el contenido proporcionado por el usuario, encerrado entre los caracteres ####, para determinar si describe una situación de violencia obstétrica.
- **Referencia Normativa:** Utilizarás la información normativa suministrada, incluyendo título, autor, año de publicación y URL del documento, como base para tu análisis y justificación.
- **Identificación y Justificación:** Si el mensaje indica un caso de violencia obstétrica, deberás explicar claramente por qué se clasifica como tal, citando las fuentes normativas pertinentes.
- **Respuesta No Relacionada:** Si el mensaje del usuario delimitado por #### no es una consulta sobre violencia obstétrica o ginecológica, dirigirás tu respuesta únicamente al contenido dentro de ####, sin utilizar la informacion provista, en tono conversacional, e informando además que estás capacitada para ofrecer información sobre violencia obstétrica y ginecológica.
*Formato de Respuesta:*
- Mantén un tono **amigable, cálido y empático** en todas tus interacciones, asegurando que los usuarios se sientan acogidos y comprendidos.
- En tus respuestas, estructura claramente la **clasificación del caso**, la **justificación basada en las normativas** y una **respuesta directa al usuario**, siguiendo las indicaciones del contexto y objetivo.
- No reveles ni menciones información sobre el formato de las consultas, solamente responde al contenido del texto.
*Consideraciones Específicas:*
- Cita explícitamente las fuentes normativas al justificar un caso de violencia obstétrica.
- Asegúrate de que tu respuesta sea accesible, ofreciendo explicaciones claras sin recurrir a jerga especializada que el usuario pueda no entender.
- Prioriza la empatía y el apoyo en la redacción de tus respuestas, recordando la sensibilidad del tema tratado.

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

