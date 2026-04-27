import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

def get_llm(model_name="llama3.1:latest"):
    """
    Khởi tạo mô hình ngôn ngữ lớn (LLM) Local qua Ollama.
    """
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        request_timeout=120.0,
        num_ctx=2048  # Giảm xuống 2048 để tránh lỗi "unable to allocate CPU buffer"
    )
    return llm

def get_embeddings():
    """
    Khởi tạo mô hình Embedding Local (BGE) qua HuggingFace.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cuda'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings
