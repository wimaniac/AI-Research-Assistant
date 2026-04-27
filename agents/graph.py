import time
import uuid
import logging
from langgraph.graph import StateGraph, START, END

from core.rag_pipeline import create_retriever
from agents.state import AgentState
from agents.nodes import (
    planner, retriever_node, evaluator, query_refiner, summarizer, 
    check_sufficient, embeddings
)

logger = logging.getLogger(__name__)

# ==========================================
# XÂY DỰNG LUỒNG ĐỒ THỊ LANGGRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever_node)
workflow.add_node("evaluator", evaluator)
workflow.add_node("query_refiner", query_refiner)
workflow.add_node("summarizer", summarizer)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "evaluator")

workflow.add_conditional_edges(
    "evaluator",
    check_sufficient,
    {
        "sufficient": "summarizer",
        "insufficient": "query_refiner"
    }
)

workflow.add_edge("query_refiner", "retriever")
workflow.add_edge("summarizer", END)

# Biên dịch Tác nhân RAG
agent_app = workflow.compile()

# ==========================================
# HÀM CHẠY TÁC NHÂN (ENTRY POINT)
# ==========================================
def run_agentic_rag(query: str):
    """Hàm chạy Agent, sử dụng generator (yield) để stream trạng thái tiến trình về UI."""
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"[ReqID: {req_id}] 🚀 BẮT ĐẦU NGHIÊN CỨU TÁC NHÂN: '{query}'")
    initial_state = {
        "request_id": req_id, "user_query": query, "sub_queries": [],
        "retrieved_docs": [], "iteration_count": 0, "is_sufficient": False, "final_answer": ""
    }
    
    final_state_accumulator = initial_state.copy()
    start_time = time.time()

    for output in agent_app.stream(initial_state):
        for node_name, node_state in output.items():
            final_state_accumulator.update(node_state)
            
            msgs = {
                "planner": f"🧠 Lập kế hoạch xong: Đã tạo {len(node_state.get('sub_queries', []))} hướng tìm kiếm.",
                "retriever": f"🔎 Thu thập dữ liệu: Đang giữ {len(node_state.get('retrieved_docs', []))} đoạn tài liệu giá trị nhất.",
                "evaluator": f"⚖️ Đánh giá: {'Đã đủ thông tin' if node_state.get('is_sufficient') else 'Còn thiếu, cần tìm thêm...'}.",
                "query_refiner": f"🔄 Bắt đầu vòng lặp mở rộng tìm kiếm (Lần {node_state.get('iteration_count', 0)})...",
                "summarizer": "📝 Đang tổng hợp kiến thức và viết báo cáo cuối..."
            }
            if node_name in msgs:
                yield {"type": "progress", "msg": msgs[node_name]}
    
    total_latency = time.time() - start_time
    logger.info(f"[ReqID: {req_id}] 🏁 TỔNG THỜI GIAN CHẠY: {total_latency:.2f}s")
    final_docs = final_state_accumulator.get("retrieved_docs", [])
    final_retriever = create_retriever(final_docs, embeddings) if final_docs else None
        
    yield {
        "type": "result", "report": final_state_accumulator.get("final_answer", ""),
        "retriever": final_retriever, "request_id": req_id, "latency": total_latency
    }