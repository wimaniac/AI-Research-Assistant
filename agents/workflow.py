from core.llm import get_llm, get_embeddings
from core.search import search_web
from core.scraper import scrape_urls
from core.rag_pipeline import create_retriever
from agents.prompts import PLANNER_PROMPT, WRITER_PROMPT
from langchain_core.output_parsers import StrOutputParser


def run_research(topic: str):
    """
    Hàm chính thực thi toàn bộ quy trình nghiên cứu:
    1. Plan -> 2. Search -> 3. Scrape -> 4. Index (RAG) -> 5. Write
    """
    llm = get_llm()
    embeddings = get_embeddings()

    # --- GIAI ĐOẠN 1: LẬP KẾ HOẠCH (PLANNING) ---
    print(f"Đang lập kế hoạch nghiên cứu về: {topic}...")
    planner_chain = PLANNER_PROMPT | llm | StrOutputParser()
    plan_result = planner_chain.invoke({"topic": topic})

    # Tách kết quả thành list các câu query
    queries = [q.strip() for q in plan_result.split('\n') if q.strip()]
    print(f"Các từ khóa tìm kiếm: {queries}")

    # --- GIAI ĐOẠN 2: TÌM KIẾM & THU THẬP (RESEARCH) ---
    all_urls = []
    # Tìm kiếm cho từng query (Lấy top 2 mỗi query để không quá nhiều rác)
    for query in queries:
        search_res = search_web(query, max_results=2)
        urls = [item['link'] for item in search_res]
        all_urls.extend(urls)

    # Lọc trùng lặp URL
    all_urls = list(set(all_urls))
    print(f"Tìm thấy {len(all_urls)} đường link liên quan.")

    # Đọc nội dung (Scrape)
    if not all_urls:
        return "Xin lỗi, tôi không tìm thấy tài liệu nào trên internet về chủ đề này."

    docs = scrape_urls(all_urls)
    if not docs:
        return "Không thể đọc nội dung từ các đường link tìm được (có thể do chặn bot)."

    # --- GIAI ĐOẠN 3: XỬ LÝ RAG (INDEXING) ---
    print("Đang tổng hợp và ghi nhớ kiến thức...")
    # Sử dụng class SimpleEnsembleRetriever của bạn ở bước này
    retriever = create_retriever(docs, embeddings)

    if not retriever:
        return "Dữ liệu quá ngắn để phân tích."

    # --- GIAI ĐOẠN 4: VIẾT BÁO CÁO (WRITING) ---
    print("Đang viết báo cáo...")

    # Truy xuất thông tin quan trọng nhất từ bộ nhớ vừa tạo
    # Hỏi retriever chính chủ đề gốc để lấy context tổng quan
    relevant_docs = retriever.invoke(topic)

    # Ghép nội dung các doc lại thành 1 chuỗi context
    context_text = "\n\n".join([d.page_content for d in relevant_docs])

    # Gọi LLM viết bài
    writer_chain = WRITER_PROMPT | llm | StrOutputParser()
    final_report = writer_chain.invoke({"topic": topic, "context": context_text})

    return {
        "report": final_report,
        "retriever": retriever  # Trả về bộ nhớ để chat tiếp
    }