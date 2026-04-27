import json
import time
import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm, get_embeddings
from core.search import search_web
from core.scraper import scrape_urls
from core.rag_pipeline import create_retriever
from agents.prompts import PLANNER_PROMPT, EVALUATOR_PROMPT, WRITER_PROMPT
from agents.state import AgentState

logger = logging.getLogger(__name__)

llm = get_llm()
embeddings = get_embeddings()

search_cache = {}
scrape_cache = {}

def clear_agent_cache():
    """Xóa bộ nhớ đệm tìm kiếm để giải phóng RAM và tìm kiếm mới mẻ hơn."""
    global search_cache, scrape_cache
    search_cache.clear()
    scrape_cache.clear()
    logger.info("🧹 Đã xóa bộ nhớ đệm tìm kiếm web.")

try:
    from sentence_transformers import CrossEncoder
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    logger.info("✅ Đã tải mô hình Reranker (Cross-Encoder).")
except Exception:
    reranker_model = None
    logger.warning("⚠️ Không tìm thấy 'sentence_transformers'. Sẽ bỏ qua bước xếp hạng lại.")

def rerank_docs(query: str, docs: List[Document], top_n: int = 10) -> List[Document]:
    if not reranker_model or not docs:
        return docs[:top_n]
    
    try:
        logger.info(f"Reranker: Đang xếp hạng lại {len(docs)} tài liệu để chọn top {top_n}...")
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker_model.predict(pairs)
        
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_score_pairs[:top_n]]
    except Exception as e:
        logger.error(f"Reranker: Lỗi ({e}). Trả về kết quả ban đầu.")
        return docs[:top_n]

def planner(state: AgentState) -> AgentState:
    req_id = state.get("request_id", "UNKNOWN")
    start_time = time.time()
    logger.info(f"[ReqID: {req_id}] 🧠 [Planner] Đang lập kế hoạch cho truy vấn: '{state['user_query']}'")
    
    try:
        planner_chain = PLANNER_PROMPT | llm | StrOutputParser()
        plan_result = planner_chain.invoke({"user_query": state["user_query"]})
        
        try:
            # Tìm khối JSON bằng regex để loại bỏ text rác (ví dụ: markdown ```json ... ```)
            match = re.search(r'\[.*\]', plan_result, flags=re.DOTALL)
            if match:
                sub_queries = json.loads(match.group(0))
            else:
                sub_queries = json.loads(plan_result)
                
            if not isinstance(sub_queries, list):
                sub_queries = [str(plan_result).strip()]
        except:
            sub_queries = []
            for q in plan_result.split('\n'):
                q = q.strip()
                # Loại bỏ các câu giao tiếp ảo giác do LLM tự sinh
                if len(q) > 10 and not q.lower().startswith(("dưới đây", "hy vọng", "chắc chắn", "vâng", "đây là")):
                    q = re.sub(r'^(\d+\.|-|\*|\"|\')\s*', '', q).strip(' "\'')
                    if q:
                        sub_queries.append(q)
            if not sub_queries:
                sub_queries = [state["user_query"]]
    except Exception as e:
        logger.error(f"[ReqID: {req_id}] ❌ [Planner] Lỗi tải mô hình LLM ({e}). Sử dụng truy vấn gốc.")
        sub_queries = [state["user_query"]]
        
    latency = time.time() - start_time
    logger.info(f"[ReqID: {req_id}] ⏱️ [Planner] Hoàn thành trong {latency:.2f}s | Truy vấn con: {sub_queries}")
    return {"sub_queries": sub_queries, "iteration_count": state.get("iteration_count", 0) + 1}

def retriever_node(state: AgentState) -> AgentState:
    req_id = state.get("request_id", "UNKNOWN")
    start_time = time.time()
    
    logger.info(f"[ReqID: {req_id}] 🔎 [Retriever] Vòng lặp thứ {state['iteration_count']}. Đang tìm kiếm tài liệu...")
    sub_queries = state["sub_queries"]
    all_docs = state.get("retrieved_docs", [])
    
    urls_to_scrape = []
    for query in sub_queries:
        if query in search_cache:
            urls = search_cache[query]
            logger.info(f"[ReqID: {req_id}] ⚡ [Cache Hit] Đã lấy URLs trong bộ nhớ đệm cho: '{query}'")
        else:
            logger.info(f"[ReqID: {req_id}] 🌐 [Tavily Search] Đang tìm kiếm trên web: '{query}'")
            try:
                search_res = search_web(query, max_results=5)
                urls = [item['link'] for item in search_res if item.get('link')]
                search_cache[query] = urls
            except Exception as e:
                logger.error(f"[ReqID: {req_id}] ❌ [Tavily Search] Lỗi tìm kiếm '{query}': {e}")
                urls = []
        urls_to_scrape.extend(urls)
        
    urls_to_scrape = list(set(urls_to_scrape))
    uncached_urls = []
    cached_docs = []
    
    for url in urls_to_scrape:
        if url in scrape_cache:
            cached_docs.append(scrape_cache[url])
        else:
            uncached_urls.append(url)
            
    if cached_docs:
        logger.info(f"[ReqID: {req_id}] ⚡ [Cache Hit] Đã lấy nội dung {len(cached_docs)} trang web từ bộ nhớ đệm.")
        
    scraped_docs = []
    if uncached_urls:
        logger.info(f"📥 [Scraper] Đang cào dữ liệu từ {len(uncached_urls)} trang web mới...")
        try:
            scraped_docs = scrape_urls(uncached_urls)
            if scraped_docs:
                for doc in scraped_docs:
                    source = doc.metadata.get('source', '') if hasattr(doc, 'metadata') else ''
                    if source:
                        scrape_cache[source] = doc
        except Exception as e:
            logger.error(f"[ReqID: {req_id}] ❌ [Scraper] Bỏ qua cào dữ liệu do lỗi: {e}")
                    
    all_scraped_docs = cached_docs + scraped_docs
    if all_scraped_docs:
        temp_retriever = create_retriever(all_scraped_docs, embeddings)
        if temp_retriever:
            for query in sub_queries:
                relevant_chunks = temp_retriever.invoke(query)
                all_docs.extend(relevant_chunks)

    unique_docs_dict = {doc.page_content: doc for doc in all_docs}
    unique_docs = list(unique_docs_dict.values())
    
    if reranker_model and unique_docs:
        # Giảm số lượng tài liệu giữ lại từ 20 xuống 10 để tránh tràn RAM/VRAM của LLM
        unique_docs = rerank_docs(state["user_query"], unique_docs, top_n=10)
        
    latency = time.time() - start_time
    logger.info(f"[ReqID: {req_id}] ⏱️ [Retriever] Hoàn thành trong {latency:.2f}s | Trạng thái bộ nhớ: {len(unique_docs)} chunks.")
    return {"retrieved_docs": unique_docs}

def evaluator(state: AgentState) -> AgentState:
    req_id = state.get("request_id", "UNKNOWN")
    start_time = time.time()
    
    logger.info(f"[ReqID: {req_id}] ⚖️ [Evaluator] Đang đánh giá tính đầy đủ của thông tin...")
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        logger.warning(f"[ReqID: {req_id}] ⚠️ [Evaluator] Trống! Chưa có dữ liệu. Buộc tìm kiếm thêm.")
        return {"is_sufficient": False, "sub_queries": [state["user_query"]]}
        
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # Cắt bớt văn bản nếu quá dài (10000 ký tự ~ 2500 tokens) để bảo vệ LLM
    if len(context_text) > 10000:
        context_text = context_text[:10000] + "\n...[Nội dung đã bị cắt bớt]..."

    try:
        evaluator_chain = EVALUATOR_PROMPT | llm | StrOutputParser()
        eval_result = evaluator_chain.invoke({
            "topic": state["user_query"],
            "context": context_text
        })
        
        eval_lines = [line.strip() for line in eval_result.strip().split('\n') if line.strip()]
        decision = eval_lines[0].upper() if eval_lines else "NO"
    except Exception as e:
        logger.error(f"[ReqID: {req_id}] ❌ [Evaluator] Lỗi LLM: {e}. Ép buộc đi tiếp.")
        eval_lines = []
        decision = "NO" # Giả định là đủ để không bị kẹt trong vòng lặp lỗi
    
    is_sufficient = "NO" in decision
    new_queries = eval_lines[1:] if not is_sufficient and len(eval_lines) > 1 else []
        
    logger.info(f"📝 [Evaluator] Kết luận: {'Đã Đủ' if is_sufficient else 'Còn Thiếu'}")
    if not is_sufficient:
        logger.info(f"🔄 [Evaluator] Đề xuất tìm thêm: {new_queries}")
        
    return {"is_sufficient": is_sufficient, "sub_queries": new_queries}

def query_refiner(state: AgentState) -> AgentState:
    req_id = state.get("request_id", "UNKNOWN")
    logger.info(f"[ReqID: {req_id}] 🔧 [Query Refiner] Đang làm mới tham số trước vòng lặp thứ {state['iteration_count'] + 1}...")
    return {"iteration_count": state["iteration_count"] + 1}

def summarizer(state: AgentState) -> AgentState:
    req_id = state.get("request_id", "UNKNOWN")
    start_time = time.time()
    
    logger.info(f"[ReqID: {req_id}] 📝 [Summarizer] Bắt đầu tổng hợp kiến thức và viết báo cáo cuối...")
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        logger.warning(f"[ReqID: {req_id}] ⚠️ [Summarizer] Không có dữ liệu đầu vào. Trả về thông báo lỗi an toàn.")
        final_answer = "Xin lỗi, hệ thống không thể tìm thấy hoặc trích xuất được bất kỳ thông tin nào từ Internet về chủ đề này. Vui lòng thử một chủ đề khác hoặc thay đổi từ khóa tìm kiếm."
        return {"final_answer": final_answer}

    context_parts = []
    for d in docs:
        source = d.metadata.get('source', '') if hasattr(d, 'metadata') and d.metadata else ''
        if source:
            context_parts.append(f"[Nguồn: {source}]\n{d.page_content}")
        else:
            context_parts.append(d.page_content)
            
    context_text = "\n\n".join(context_parts)
    
    # Cắt bớt văn bản nếu quá dài (15000 ký tự ~ 4000 tokens)
    if len(context_text) > 15000:
        context_text = context_text[:15000] + "\n...[Nội dung đã bị cắt bớt]..."
    
    try:
        writer_chain = WRITER_PROMPT | llm | StrOutputParser()
        final_answer = writer_chain.invoke({
            "topic": state["user_query"],
            "context": context_text
        })
    except Exception as e:
        logger.error(f"[ReqID: {req_id}] ❌ [Summarizer] Lỗi LLM: {e}")
        final_answer = f"⚠️ Rất tiếc, mô hình AI Local đã gặp sự cố (quá tải bộ nhớ) khi viết báo cáo. Vui lòng thử lại với từ khóa hẹp hơn."
    
    used_urls = set()
    for d in docs:
        source = d.metadata.get('source', '') if hasattr(d, 'metadata') and d.metadata else ''
        if source:
            used_urls.add(source)
            
    if used_urls:
        # Cắt bỏ phần Nguồn tham khảo do LLM tự sinh (vì AI nội bộ thường quên format link)
        match = re.search(r'\n[ \t]*(#+|\*\*|__)?\s*(🔗\s*)?(nguồn tham khảo|tài liệu tham khảo|references?|sources?)', final_answer, flags=re.IGNORECASE)
        if match:
            final_answer = final_answer[:match.start()].strip()
            
        # Luôn luôn tự động nối link chuẩn Markdown (có thể click) bằng Python
        final_answer += "\n\n## 🔗 Nguồn tham khảo\n"
        for url in used_urls:
            final_answer += f"* [{url}]({url})\n"

    latency = time.time() - start_time
    logger.info(f"[ReqID: {req_id}] ⏱️ [Summarizer] Hoàn thành trong {latency:.2f}s | ✅ Đã hoàn tất báo cáo!")
    return {"final_answer": final_answer}

def check_sufficient(state: AgentState) -> str:
    req_id = state.get("request_id", "UNKNOWN")
    max_iterations = 3
    if state["is_sufficient"]:
        return "sufficient"
    elif state["iteration_count"] >= max_iterations:
        logger.warning(f"[ReqID: {req_id}] ⚠️ [Condition Edge] Đã đạt giới hạn lặp ({max_iterations}). Buộc chuyển sang tổng hợp!")
        return "sufficient"
    else:
        return "insufficient"