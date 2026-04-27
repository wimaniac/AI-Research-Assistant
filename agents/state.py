from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    """Định nghĩa trạng thái của Agentic RAG Workflow"""
    request_id: str
    user_query: str
    sub_queries: List[str]
    retrieved_docs: List[Document]
    iteration_count: int
    final_answer: str
    is_sufficient: bool