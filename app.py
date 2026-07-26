import streamlit as st

from chatbot.rag_chatbot import RAGChatbot


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="E-Commerce RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 E-Commerce RAG Chatbot")
st.write("Ask questions about the uploaded documents.")


# ----------------------------
# Initialize Chatbot.
# ----------------------------

@st.cache_resource
def load_chatbot():
    return RAGChatbot()


chatbot = load_chatbot()


# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Display Chat History
# ----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ----------------------------
# User Input
# ----------------------------

question = st.chat_input("Ask a question...")

if question:

    # Show user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = chatbot.ask(question)

            answer = result["answer"]

            st.markdown(answer)

            with st.expander("Retrieved Chunks"):

                for i, doc in enumerate(result["documents"], start=1):

                    st.markdown(f"### Chunk {i}")

                    st.write(doc.page_content)

                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )