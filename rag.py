import os
import re

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Rag():

    def __init__(self):
        self.chunk_size = 800
        self.chunk_overlap = self.chunk_size * 0.20
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini"
        )
        self.vector_db = Chroma(
            collection_name="bitcoin",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )


    def clean_text(self, text: str):
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def load_documents(self) -> List[Document]:
        document_folders = ["byd", "mini", "opel", "peugeot", "renault", "vw"]

        all_documents = []
        for folder in document_folders:
            for i in range(1, 6):
                file_path = os.path.join(f"documents/{folder}/review_{i}.pdf")
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                for page in pages:
                    page.page_content = self.clean_text(page.page_content)
                all_documents.extend(pages)

        return all_documents

    def split_docs(self, documents: List[Document]) -> List[Document]:
        # Phase 2 Document Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        splitted_docs = text_splitter.split_documents(documents)
        return splitted_docs

    def store_on_vector_db_splitted_docs(self, splitted_docs):
        # Phase 3 Store on vector
        self.vector_db.add_documents(splitted_docs)

    def database_search(self, query, k=5):
        relevant_docs = self.vector_db.similarity_search(
            query,
            k=k,
        )
        return relevant_docs


    def inference_phase(self, message: str, history):
        query = message
        print(query)

        # Phase 4 similarity search on vector db to find most relevant docs
        relevant_docs = self.database_search(query)

        # Creating Context
        context = ""
        for doc in relevant_docs:
            context += doc.page_content + '\n'

        # Creating final system prompt
        system_prompt = """
            You are an assistant for question-answering tasks.
            Use only the following pieces of retrieved context from a document to answer the user question.
            If the document doesn't provide the answer, tell the user you don't know that information.
            If you don't find the information, try to search again on the provided context.
            The document is about cars reviews.
            The reviews that you have are: Vauxhall Astra Electric Sports Tourer, BYD Seal, Mini Cooper, Peugeot 308, Renault Megane E-Tech Electric, Volkswagen ID.5 
            Return the filename as a reference when you use it
            In the end respond always with something funny

            **Context**
            {context}
            ** End of Context**

            **User query**
            {query}
            **End of user query**
        """
        system_prompt = system_prompt.replace("{context}", context).replace("{query}", query)

        # "human / user / chatgpt e metemos uma mensagem", "assistant / llm / resposta llm", "system / backend / hardcoded message on backend systemPromptPT"
        messages = []
        count = 0
        if len(history) > 0:
            for message in history[0]:
                print(message)
                if count % 2 == 0:
                    messages.append({"role": "human", "content": message})
                else:
                    messages.append({"role": "assistant", "content": message})
                count = count + 1

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "human", "content": query})

        response = self.llm.invoke(messages)
        return response.content

    def create(self):
        # Phase 1 load documents
        documents = self.load_documents()
        splitted_docs = self.split_docs(documents)
        self.store_on_vector_db_splitted_docs(splitted_docs)
