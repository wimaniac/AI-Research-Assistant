from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5):
    print(f"🔎 Đang tìm kiếm: {query}...")
    results = []

    try:
        # Sử dụng backend='lite' hoặc 'html' giúp tránh bị chặn IP trên Cloud
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results, backend="lite")

            if search_results:
                for r in search_results:
                    # --- [FIX QUAN TRỌNG]: Kiểm tra xem r có phải là dict không ---
                    if isinstance(r, str):
                        print(f"⚠️ Dữ liệu rác (String): {r}")
                        continue
                    # -------------------------------------------------------------

                    results.append({
                        "title": r.get('title', ''),
                        "link": r.get('href', ''),
                        "snippet": r.get('body', '')
                    })
    except Exception as e:
        print(f"⚠️ Lỗi khi tìm kiếm '{query}': {e}")
        # Nếu backend='lite' lỗi, có thể thử fallback sang backend='html' ở đây nếu muốn

    return results