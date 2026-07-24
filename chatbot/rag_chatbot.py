from langchain_chroma import Chroma

from config import (
    CHROMA_DB,
    TOP_K,
)

from services.embeddings import get_embedding_model
from services.llm import get_llm


PROMPT_TEMPLATE = """
You are an AI assistant for an e-commerce website.

Instructions:
- Answer ONLY using the provided context.
- If the answer is not found in the context, say:
"I couldn't find that information in the provided documents."
- Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""


class RAGChatbot:

    def __init__(self):
        """
        Initialize the chatbot.
        This runs only once when the application starts.
        """

        print("Loading embedding model...")
        self.embeddings = get_embedding_model()

        print("Loading Chroma database...")
        self.vector_db = Chroma(
            persist_directory=CHROMA_DB,
            embedding_function=self.embeddings,
        )

        print("Creating retriever...")
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": TOP_K}
        )

        print("Loading Groq model...")
        self.llm = get_llm()

        print("Chatbot is ready.\n")

    def retrieve_documents(self, question):
        """
        Retrieve relevant chunks.
        """

        return self.retriever.invoke(question)

    def build_context(self, documents):
        """
        Convert retrieved chunks into one context string.
        """

        return "\n\n".join(
            doc.page_content
            for doc in documents
        )

    def generate_answer(self, question, context):
        """
        Generate answer from the LLM.
        """

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        response = self.llm.invoke(prompt)

        return response.content

    def ask(self, question):
        """
        Main chatbot function.
        """

        try:

            documents = self.retrieve_documents(question)

            context = self.build_context(documents)

            answer = self.generate_answer(
                question,
                context,
            )

            return {
                "answer": answer,
                "documents": documents,
                "context": context,
            }

        except Exception as e:

            return {
                "answer": f"Error: {e}",
                "documents": [],
                "context": "",
            }