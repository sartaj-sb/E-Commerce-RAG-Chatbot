from chatbot.rag_chatbot import RAGChatbot

chatbot = RAGChatbot()

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    result = chatbot.ask(question)

    print("\nAnswer\n")
    print(result["answer"])

    print("\nRetrieved Chunks\n")

    for i, doc in enumerate(result["documents"], start=1):

        print(f"\nChunk {i}")

        print("-" * 60)

        print(doc.page_content)

        print("-" * 60)