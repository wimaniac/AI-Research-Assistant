import json
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any

from agents.graph import run_agentic_rag
from agents.nodes import clear_agent_cache
from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="AI Research Assistant API")

# --- CONCURRENCY CONTROL ---
# Chuyển Lock xuống Backend để bảo vệ mô hình Local LLM khỏi bị quá tải
GLOBAL_AGENT_LOCK = threading.Lock()

# --- MEMORY STORE ---
# Lưu trữ bộ tìm kiếm (Retriever) trên RAM của Backend cho từng phiên để hỗ trợ chat
session_retrievers: Dict[str, Any] = {}

class ResearchRequest(BaseModel):
    topic: str

class ChatRequest(BaseModel):
    request_id: str
    question: str

@app.post("/research")
async def research(req: ResearchRequest):
    if not GLOBAL_AGENT_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Hệ thống đang bận xử lý một yêu cầu khác. Vui lòng thử lại sau!")

    def generate():
        try:
            for update in run_agentic_rag(req.topic):
                if update["type"] == "result":
                    # Lưu retriever vào bộ nhớ của Backend, KHÔNG gửi qua HTTP
                    req_id = update.get("request_id")
                    if req_id and update.get("retriever"):
                        session_retrievers[req_id] = update["retriever"]
                    
                    # Bỏ object phức tạp khỏi response JSON
                    update.pop("retriever", None)
                
                # NDJSON (Newline Delimited JSON) format for streaming
                yield json.dumps(update) + "\n"
        finally:
            # Luôn nhả Lock ngay cả khi client ngắt kết nối đột ngột
            GLOBAL_AGENT_LOCK.release()

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/chat")
async def chat(req: ChatRequest):
    retriever = session_retrievers.get(req.request_id)
    if not retriever:
        return {"answer": "⚠️ Xin lỗi, phiên làm việc đã hết hạn hoặc không tìm thấy dữ liệu. Vui lòng nghiên cứu lại."}
    
    llm = get_llm()
    related_docs = retriever.invoke(req.question)
    context_text = "\n\n".join([d.page_content for d in related_docs])
    
    chat_prompt = ChatPromptTemplate.from_template("""
    Bạn là trợ lý nghiên cứu. Người dùng đang hỏi về báo cáo đã tạo.
    Dữ liệu liên quan tìm thấy (Context):
    {context}
    Câu hỏi của người dùng: {question}
    Hãy trả lời ngắn gọn, súc tích dựa trên Context trên. 
    Nếu không có thông tin trong Context, hãy nói là "Dữ liệu thu thập được chưa đề cập đến vấn đề này".
    """)
    
    chain = chat_prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": req.question})
    return {"answer": response}

@app.post("/clear_cache")
async def clear_cache():
    clear_agent_cache()
    session_retrievers.clear()
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)