import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (
    PDF_FOLDER,
    CHROMA_DB,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from services.embeddings import get_embedding_model


def load_documents():
    """
    Load all PDF files from the data folder.
    """
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vector_store(chunks):
    """
    Create a fresh Chroma vector database.
    Deletes the old database first to avoid duplicate embeddings.
    """

    if os.path.exists(CHROMA_DB):
        shutil.rmtree(CHROMA_DB)
        print("Existing Chroma database deleted.")

    embeddings = get_embedding_model()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB,
    )

    print("Vector database created successfully.")


def main():

    print("\n========== INGESTION STARTED ==========\n")

    print("Loading PDF documents...")
    documents = load_documents()

    if not documents:
        print("No PDF files found in the data folder.")
        return

    print("\nSplitting documents...")
    chunks = split_documents(documents)

    if not chunks:
        print("No chunks were created.")
        return

    print("\nCreating vector database...")
    create_vector_store(chunks)

    print("\n========== INGESTION COMPLETED ==========\n")


if __name__ == "__main__":
    main()