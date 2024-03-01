from pypdf import PdfReader
import tiktoken
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Constants import *
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

tokenizer = tiktoken.get_encoding("cl100k_base")

def pdf_to_string(loaded_doc):
    document_text = ""
    for page in loaded_doc.pages:
        document_text += page.extract_text()

    return document_text

def token_counter(text):
    return len(tokenizer.encode(text))

openai_embedding = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8192,
    chunk_overlap=100,
    length_function=token_counter,
    separators=["\n\n", ".", "\n", " "]
)

def main():
    doc_list = os.listdir("../datos")
    doc_list = [doc for doc in doc_list if doc != ".ipynb_checkpoints"]
    corpus = []
    for index, doc_path in enumerate(doc_list):
        print(f"loading document {index + 1} of {len(doc_list)}")
        loaded_doc = PdfReader(f"../datos/{doc_path}")
        document_text = pdf_to_string(loaded_doc)
        corpus += [{"text": document_text}]

    chunks = text_splitter.create_documents(
        texts = [document["text"] for document in corpus]
    )

    chroma_client = PersistentClient('chroma')

    collection = chroma_client.create_collection(
        name='vo-normas',
        embedding_function=openai_embedding,
        metadata={"hnsw:space": "ip"}
    )

    collection.add(
        documents=[document.page_content for document in chunks],
        ids=[f'id{i + 1}' for i in range(len(chunks))]
    )

if __name__ == '__main__':
    main()
