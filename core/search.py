import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


def search_web(query: str, max_results: int = 5):
    """
    Tìm kiếm sử dụng Tavily API (Chuyên dụng cho AI Agent).
    Ổn định hơn DuckDuckGo và không bị chặn IP trên Cloud.
    """
    print(f"🔎 Đang tìm kiếm (Tavily): {query}...")

    if "TAVILY_API_KEY" not in os.environ:
        print("⚠️ Lỗi: Chưa cấu hình TAVILY_API_KEY trong file .env hoặc chưa load được file.")
        return []

    try:
        # Khởi tạo công cụ Tavily
        # search_depth="advanced" giúp tìm sâu hơn, nhưng "basic" thì nhanh hơn.
        tool = TavilySearchResults(max_results=max_results)

        # Gọi API
        raw_results = tool.invoke({"query": query})

        # Chuẩn hóa dữ liệu đầu ra cho khớp với format cũ của dự án
        # Tavily trả về: [{'url': '...', 'content': '...'}]
        results = []
        for r in raw_results:
            results.append({
                # Tavily tập trung vào content, ít khi trả title riêng,
                # nên ta lấy 60 ký tự đầu của content làm title tạm.
                "title": r.get('content', '')[:60] + "...",
                "link": r.get('url', ''),
                "snippet": r.get('content', '')
            })

        return results

    except Exception as e:
        print(f"⚠️ Lỗi khi gọi Tavily API: {e}")
        return []


