import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY chưa được cấu hình trong file .env")

def get_llm(model_name="gemini-1.5-flash"):
    """
    Khởi tạo mô hình Google Gemini.
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        max_output_tokens=4096,
        convert_system_message_to_human=True
    )
    return llm

def get_embeddings():
    """
    Khởi tạo mô hình Embedding của Google để dùng cho Vector Store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )
    return embeddings
