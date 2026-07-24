from langchain_chroma import Chroma

from config import (
    CHROMA_DB,
    TOP_K,
)

from services.embeddings import get_embedding_model


def load_vector_store():
    """
    Load the existing Chroma vector database.
    """

    embeddings = get_embedding_model()

    vector_db = Chroma(
        persist_directory=CHROMA_DB,
        embedding_function=embeddings,
    )

    return vector_db


def get_retriever():
    """
    Create and return a retriever.
    """

    vector_db = load_vector_store()

    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    return retriever


def retrieve_documents(query):
    """
    Retrieve the most relevant document chunks.
    """

    retriever = get_retriever()

    documents = retriever.invoke(query)

    return documents


def print_documents(documents):
    """
    Print retrieved chunks.
    """

    print("\n" + "=" * 80)

    print(f"Retrieved {len(documents)} chunks")

    print("=" * 80)

    for index, document in enumerate(documents, start=1):

        print(f"\nChunk {index}")

        print("-" * 80)

        print(document.page_content)

        print("-" * 80)


def main():

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        documents = retrieve_documents(query)

        print_documents(documents)


if __name__ == "__main__":
    main()