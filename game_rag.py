import glob
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import chromadb


# load_dotenv()
# anthropic_api = os.getenv("ANTHROPIC_API_KEY")

llm = OpenAI(api_key = "ollama", base_url='http://host.docker.internal:11434/v1')

MODEL = "llama3.2"

loader = DirectoryLoader("knowledge-base/games", glob = "**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("chroma_db"):
    import shutil
    shutil.rmtree("chroma_db")


vector_store = Chroma(collection_name = "games", embedding_function=embeddings, persist_directory="chroma_db")
vector_store.add_documents(documents=chunks)

print(f"Total chunks in DB: {vector_store._collection.count()}")

def rerank(question, docs):
    # Промпт
    prompt = f"Question: {question}\n\n Sort in relevance order.\n\n"
    for i, doc in enumerate(docs):
        prompt += f"ID {i+1}: {doc.page_content[:200]}\n\n"
    prompt += "Return in JSON format! It's QUITLY important!!!!"
    prompt += "\n Make a list only with ID's of your answers like [1,2,4,5]! Return only this list, don't add any words before or after the output"


    # Запит
    response = llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    print("LLM відповів:", answer)  # ← тут дебаг

    # Парсинг
    order = json.loads(response.choices[0].message.content)
    
    # Повернення
    return [docs[i-1] for i in order]


def chat(message, history):
    system_message = "You are game assistant who helps people find some information about games"

    results = vector_store.similarity_search(message, k = 8)
    results = rerank(message, results)[:5]  
    res_text = "\n".join([doc.page_content for doc in results])
    
    history = [{"role":h["role"], "content":h["content"]} for h in history]

    system_with_res = system_message + " Finded context: " + res_text

    messages = [{"role" : "system", "content" : system_with_res}] + history + [{"role" : "user", "content" : message}]

    stream = llm.chat.completions.create(model=MODEL, messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat).launch(inbrowser = True, share = True)
