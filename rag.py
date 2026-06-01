from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import settings
from typing import List
from langchain_core.documents import Document
from ingest import texts
from langchain_anthropic import ChatAnthropic


embeddings = HuggingFaceEmbeddings(model=settings.embedded_model)

vector_store = Chroma(
    collection_name="games_knowledge",
    embedding_function=embeddings,
    persist_directory=settings.chroma_persist_dir
)

def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def build_prompt():
    system_message = "You are game assistant who helps people find some information about games"
    retriever_res = retrieve_context(settings.message)[0]
    system_prompt = system_message + " " + retriever_res
    messages = [{"role" : "system", "content" : system_prompt}, {"role" : "human", "content" : settings.message}]
    return messages

model = ChatAnthropic(
    model = settings.anthropic_model,
    temperature=0,
    api_key=settings.anthropic_api_key
)

messages = build_prompt()

for chunk in model.stream(messages):
    print(chunk.content, end="")