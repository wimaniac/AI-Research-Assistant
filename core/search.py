from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5):
    """
    Tìm kiếm trên DuckDuckGo sử dụng thư viện gốc DDGS để ổn định hơn.
    """
    print(f"🔎 Đang tìm kiếm: {query}...")
    results = []

    try:
        # Sử dụng context manager để quản lý phiên làm việc
        with DDGS() as ddgs:
            # backend="api" hoặc "html" hoặc "lite" thường ổn định
            # Hàm .text() trả về generator
            search_results = ddgs.text(query, max_results=max_results)

            if search_results:
                for r in search_results:
                    results.append({
                        "title": r.get('title', ''),
                        # Thư viện gốc trả về 'href', ta đổi thành 'link' cho đồng bộ code cũ
                        "link": r.get('href', ''),
                        # Thư viện gốc trả về 'body', ta đổi thành 'snippet'
                        "snippet": r.get('body', '')
                    })
    except Exception as e:
        print(f"⚠️ Lỗi khi tìm kiếm '{query}': {e}")

    return results


if __name__ == "__main__":
    # Test nhanh
    res = search_web("LangChain tutorial python", 3)
    print(f"Tìm thấy {len(res)} kết quả.")
    for r in res:
        print(f"- {r['title']}: {r['link']}")