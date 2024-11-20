import gradio as gr

from rag import Rag

# import os
# import re
#
# from typing import List
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# def clean_text(text: str):
#     text = re.sub(r'\n+', '\n', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text
#
#
# def load_documents() -> List:
#     document_folders = ["byd", "mini", "opel", "peugeot", "renault", "vw"]
#
#     all_documents = []
#     for folder in document_folders:
#         for i in range(1, 6):
#             file_path = os.path.join(f"documents/{folder}/review_{i}.pdf")
#             loader = PyPDFLoader(file_path)
#             pages = loader.load()
#
#             for page in pages:
#                 page.page_content = clean_text(page.page_content)
#             all_documents.extend(pages)
#
#     return all_documents
#
#
# def start():
#     # Press the green button in the gutter to run the script.
#     # Phase 1 load documents
#     documents = load_documents()
#
#     # Phase 2 Document Split
#     chunk_size = 800
#     chunk_overlap = chunk_size * 0.20
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#     splitted_docs = text_splitter.split_documents(documents)
#
#     # Phase 3 Store on vector
#     # Phase 3.1 Embed docs
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#
#     # Phase 3.2 Store on vector DB
#     vector_db = Chroma(
#         collection_name="bitcoin",
#         embedding_function=embeddings,
#         persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
#     )
#     vector_db.add_documents(splitted_docs)
#
#
# def inference_phase(message, history):
#     query = message
#     print(query)
#
#     vector_db = Chroma(
#         collection_name="bitcoin",
#         embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
#         persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
#     )
#
#     #Phase 4 similarity search on vector db to find most relevant docs
#     relevant_docs = vector_db.similarity_search(
#         query,
#         k=5,
#     )
#
#     # Creating Context
#     context = ""
#     for doc in relevant_docs:
#         context += doc.page_content + '\n'
#
#     #Creating final system prompt
#     systemPrompt = """
#         You are an assistant for question-answering tasks.
#         Use only the following pieces of retrieved context from a document to answer the user question.
#         If the document doesn't provide the answer, tell the user you don't know that information.
#         If you don't find the information, try to search again on the provided context.
#         The document is about cars reviews.
#         The reviews that you have are: Vauxhall Astra Electric Sports Tourer, BYD Seal, Mini Cooper, Peugeot 308, Renault Megane E-Tech Electric, Volkswagen ID.5
#         Return the filename as a reference when you use it
#         In the end respond always with something funny
#
#         **Context**
#         {context}
#         ** End of Context**
#
#         **User query**
#         {query}
#         **End of user query**
#     """
#     systemPrompt = systemPrompt.replace("{context}", context).replace("{query}", query)
#
#     # Objetivo final - finalPrompt = query + context + systemPrompt
#
#     # Phase 5 call the LLM with the provided prompt
#     llm = ChatOpenAI(
#         model="gpt-4o-mini"
#     )
#
#     # "human / user / chatgpt e metemos uma mensagem", "assistant / llm / resposta llm", "system / backend / hardcoded message on backend systemPromptPT"
#     messages = []
#     count = 0
#     if len(history) > 0:
#         for message in history[0]:
#             print(message)
#             if count % 2 == 0:
#                 messages.append({"role": "human", "content": message})
#             else:
#                 messages.append({"role": "assistant", "content": message})
#             count = count + 1
#
#     messages.append({"role": "system", "content": systemPrompt})
#     messages.append({"role": "human", "content": query})
#
#     response = llm.invoke(messages)
#     print(response)
#
#     return response.content


# ingestionPhase()
new_rag = Rag()
new_rag.create()
gr.ChatInterface(new_rag.inference_phase, chatbot=gr.Chatbot(value=[[None, "Hi how can i help you?"]])).launch(debug=True)