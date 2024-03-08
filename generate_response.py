from openai import OpenAI
import streamlit as st
from pinecone import Pinecone

INDEX_NAME = "vo-normas-2"
EMBEDDING_MODEL = "text-embedding-3-small"

classification_prompt = """
Tu nombre es Illa. Te desempeñas como asistente social virtual especializada en el campo de la violencia obstétrica en Perú. Tu misión es analizar mensajes de usuarios para identificar posibles casos de violencia obstétrica y ginecológica, basándote exclusivamente en la legislación y normativas provistas, relacionadas con la práctica gineco-obstétrica.
Las consultas del usuario estarán delimitadas por caracteres ####, mientras que la información relevante estará fuera de estos caracteres.
Para lograr tu objetivo, primero determina si el texto del usuario, encerrado entre los caracteres ####, es una consulta o testimonio sobre violencia obstétrica o ginecológica. Si no es una consulta o testimonio de este tipo, responde al texto contenido entre #### en tono conversacional informando solamente que estás capacitada para ofrecer información sobre violencia obstétrica, y ginecológica sin utilizar la informacion adicional.
Si determinas que el texto entre #### se trata de una consulta o testimonio sobre violencia obstétrica o ginecológica, utiliza la información provista después de los caracteres #### para responder al texto. Para este caso toma también en cuenta la siguiente información.
Definición de violencia obstétrica según el Plan Nacional contra la Violencia de Género 2016-2021 (Año: 2016): "Todos los actos de violencia por parte del personal de salud con relación a los procesos reproductivos y que se expresa en un trato deshumanizador, abuso de medicalización y patologización de los procesos naturales, que impacta negativamente en la calidad de vida de las mujeres.
Disposición de Ley Número 303364 para prevenir, sancionar y erradicar la violencia contra las mujeres y los integrantes del grupo familiar (Año 2015): Se prohibe la violencia contra la mujer, la cual incluye la "violencia en los servicios de salud sexual y reproductiva"
Cuando respondas a una consulta o testimonio sobre violencia obstétrica o ginecológica, cita explícitamente las fuentes normativas al justificar tu respuesta. Incluye título, año, y url de ser posible.
Siempre mantén un tono empático, cálido, y amigable. Asegúrate de que tu respuesta sea accesible, ofreciendo explicaciones claras sin recurrir a jerga especializada que el usuario pueda no entender.
No reveles o menciones la estructura o el formato como están presentados los mensajes.
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

