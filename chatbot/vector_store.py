from pypdf import PdfReader
import tiktoken
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Constants import *
from pinecone import Pinecone, PodSpec
from openai import OpenAI, embeddings
from create_embeddings import ENCODING_FORMAT
import json
import itertools

BATCH_SIZE = 50
openai_client = OpenAI(api_key = OPENAI_API_KEY)

def batches_generator(vectors, batch_size):
        iterable_vectors = iter(vectors)
        batch = tuple(itertools.islice(iterable_vectors, batch_size))
        while batch:
                yield batch
                batch = tuple(itertools.islice(iterable_vectors, batch_size))


def main():
        print("Loading Vectors")
        with open("embeddings.json", "r", encoding=ENCODING_FORMAT) as f:
                vectors = json.load(f)
        print("Initializing Pinecone client")
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

        print("Creating Index")
        if "vo-normas" in pinecone_client.list_indexes().names():
                pinecone_client.delete_index("vo-normas")
                
        pinecone_client.create_index(
                name="vo-normas",
                dimension=1536,
                metric="dotproduct",
                spec=PodSpec(
                environment="gcp-starter"
                )
        )
        index = pinecone_client.Index("vo-normas")

        print("Upserting Vectors")
        for vectors_batches in batches_generator(vectors, BATCH_SIZE):
                index.upsert(
                        vectors=list(vectors_batches)
                )
    
    

if __name__ == '__main__':
        main()
