from config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

directory = settings.knowledge_dir

loader = DirectoryLoader(
    directory, 
    glob = "**/*.md", 
    loader_cls=TextLoader, 
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size, 
    chunk_overlap=settings.chunk_overlap
)

texts = text_splitter.split_documents(documents)
print(texts)

embeddings = HuggingFaceEmbeddings(model=settings.embedded_model)

vector_store = Chroma(
    collection_name="games_knowledge",
    embedding_function=embeddings,
    persist_directory=settings.chroma_persist_dir
)

uuids = [str(uuid4()) for _ in range(len(texts))]
vector_store.add_documents(documents=texts, ids=uuids)



