from core.llm import get_llm, get_embeddings
from core.search import search_web
from core.scraper import scrape_urls
from core.rag_pipeline import create_retriever
from agents.prompts import PLANNER_PROMPT, WRITER_PROMPT, EVALUATOR_PROMPT
from langchain_core.output_parsers import StrOutputParser


def run_research(topic: str, max_iterations: int = 3):
    """
    Hàm chính thực thi quy trình Agentic RAG:
    1. Plan -> 2. Search & Retrieve (Loop) -> 3. Evaluate (Loop) -> 4. Synthesize
    """
    llm = get_llm()
    embeddings = get_embeddings()

    # --- BƯỚC 1: LẬP KẾ HOẠCH (PLANNING) ---
    print(f"Đang lập kế hoạch nghiên cứu về: {topic}...")
    planner_chain = PLANNER_PROMPT | llm | StrOutputParser()
    plan_result = planner_chain.invoke({"topic": topic})

    queries_to_process = [q.strip() for q in plan_result.split('\n') if q.strip()]
    print(f"Các từ khóa tìm kiếm ban đầu: {queries_to_process}")

    all_scraped_docs = []
    collected_context_chunks = []
    iteration = 0

    # --- BƯỚC 2 & 3: VÒNG LẶP TÌM KIẾM VÀ ĐÁNH GIÁ ---
    while queries_to_process and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Vòng lặp thu thập thứ {iteration}/{max_iterations} ---")
        
        for query in queries_to_process:
            print(f"🔎 Đang xử lý sub-question: {query}")
            search_res = search_web(query, max_results=2)
            urls = [item['link'] for item in search_res if item.get('link')]
            
            if urls:
                urls = list(set(urls)) # Lọc trùng
                docs = scrape_urls(urls)
                if docs:
                    all_scraped_docs.extend(docs)
                    # Tạo retriever tạm để rút trích top-k chunks cho query này
                    temp_retriever = create_retriever(docs, embeddings)
                    if temp_retriever:
                        relevant_chunks = temp_retriever.invoke(query)
                        # Lưu vào memory
                        collected_context_chunks.extend(relevant_chunks)

        # Nếu không có chunks nào thì dừng
        if not collected_context_chunks:
            return {"report": "Không thu thập được thông tin nào từ Internet.", "retriever": None}

        # Gộp context hiện tại để đánh giá
        # Loại bỏ các chunk trùng lặp để tiết kiệm context window
        unique_chunks = {chunk.page_content: chunk for chunk in collected_context_chunks}.values()
        current_context_text = "\n\n".join([d.page_content for d in unique_chunks])

        # --- BƯỚC 4: ĐÁNH GIÁ (EVALUATOR) ---
        print("⚖️ Đang đánh giá độ đầy đủ của thông tin...")
        evaluator_chain = EVALUATOR_PROMPT | llm | StrOutputParser()
        eval_result = evaluator_chain.invoke({
            "topic": topic,
            "context": current_context_text
        })
        
        eval_lines = [line.strip() for line in eval_result.strip().split('\n') if line.strip()]
        decision = eval_lines[0].upper() if eval_lines else "NO"
        
        if "YES" in decision and len(eval_lines) > 1:
            print("⚠️ Phát hiện thiếu thông tin. Sinh truy vấn mới...")
            queries_to_process = eval_lines[1:]
            print(f"Các từ khóa bổ sung: {queries_to_process}")
        else:
            print("✅ Thông tin đã đầy đủ. Chuyển sang viết báo cáo.")
            queries_to_process = [] # Đã đủ, thoát vòng lặp

    # Tạo retriever tổng thể từ tất cả doc đã scrape cho chức năng Q&A sau đó
    final_retriever = None
    if all_scraped_docs:
        final_retriever = create_retriever(all_scraped_docs, embeddings)

    # --- BƯỚC 5: TỔNG HỢP VÀ VIẾT BÁO CÁO (SYNTHESIZER) ---
    print("📝 Đang tổng hợp kiến thức và viết báo cáo...")
    unique_chunks = {chunk.page_content: chunk for chunk in collected_context_chunks}.values()
    final_context_text = "\n\n".join([d.page_content for d in unique_chunks])
    
    writer_chain = WRITER_PROMPT | llm | StrOutputParser()
    final_report = writer_chain.invoke({"topic": topic, "context": final_context_text})

    return {
        "report": final_report,
        "retriever": final_retriever  # Trả về bộ nhớ đầy đủ để chat tiếp
    }