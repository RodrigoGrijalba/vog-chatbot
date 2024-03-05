from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from openai import OpenAI
import os
import json
import tiktoken
from Constants import *
from uuid import uuid4
import pandas as pd

DATA_PATH = "../datos/docs"
METADATA_PATH = "../datos"
METADATA_FIELDS = {
        "title": "TÍTULO",
        "author": "ENTIDAD",
        "year": "AÑO",
        "url": "ENLACE"
}
tokenizer = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = "text-embedding-3-small"
openai_client = OpenAI(api_key=OPENAI_API_KEY)
ENCODING_FORMAT = "utf-8-sig"
metadata = pd.read_csv(f"{METADATA_PATH}/readables_metadata.csv")


def token_counter(text):
    return len(tokenizer.encode(text))


text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", "\n", " "],
        chunk_size=8192,
        chunk_overlap=100,
        length_function=token_counter
)

def document_to_text(document):
        document_text = ""
        for page in document.pages:
                document_text += page.extract_text()
        return document_text
        

def entries_from_path(path):
        entries = []
        document_file_name = path.split("/")[-1]
        document = PdfReader(path)
        document_text = document_to_text(document)
        path_metadata_row = metadata[metadata.NOMBRE_ARCHIVO == document_file_name]
        document_metadata = {key: str(path_metadata_row[METADATA_FIELDS[key]].values[0]) for key in METADATA_FIELDS.keys()}
        return {"text": document_text, "metadata": document_metadata}

def join_embeddings_chunks(chunks, embeddings):
        print("Joining documents and embeddings...")
        chunks_as_dict = [chunk.metadata for chunk in chunks]
        for chunk, metadata in zip(chunks, chunks_as_dict):
                metadata["text"] = chunk.page_content
        embeddings_with_metadata = [
              {"values": embed, "metadata": chunk_metadata, "id": str(uuid4())}
              for embed, chunk_metadata in zip(embeddings, chunks_as_dict)
       ]
        return embeddings_with_metadata

def embeddings_from_chunks(chunks):
        print("Embedding Documents...")
        embeddings_response = openai_client.embeddings.create(
              input=[chunk.page_content for chunk in chunks],
              model=EMBEDDING_MODEL
        )
        embeddings = [entry.embedding for entry in embeddings_response.data]
        embeddings_with_metadata = join_embeddings_chunks(chunks, embeddings)
        return embeddings_with_metadata

def main():
        doc_paths = os.listdir(DATA_PATH)
        doc_paths = [f"{DATA_PATH}/{doc}" for doc in doc_paths]
        print("Reading documents...")
        corpus_texts = [entries_from_path(path) for path in doc_paths]
        chunks = text_splitter.create_documents(
               texts=[entry["text"] for entry in corpus_texts],
               metadatas=[entry["metadata"] for entry in corpus_texts]
        )
        embed_entries = embeddings_from_chunks(chunks)
        with open("embeddings.json", "w", encoding=ENCODING_FORMAT) as f:
               json.dump(embed_entries, f)


if __name__ == "__main__":
        main()