from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5):
    print(f"🔎 Đang tìm kiếm: {query}...")
    results = []

    try:
        # Thử backend 'html' (thường ít lỗi hơn api/lite trên cloud)
        with DDGS() as ddgs:
            # Lấy raw generator
            raw_results = ddgs.text(query, max_results=max_results, backend="html")

            # Duyệt từng kết quả một cách cẩn thận
            for i, r in enumerate(raw_results):
                # DEBUG: In ra kiểu dữ liệu thực tế nhận được
                # print(f"Raw result {i}: {type(r)} - {r}")

                # CHỐT CHẶN: Nếu r là string (ví dụ thông báo lỗi), bỏ qua ngay
                if isinstance(r, str):
                    print(f"⚠️ Bỏ qua dữ liệu lỗi (String): {r}")
                    continue

                # Chỉ xử lý nếu r là dict
                if isinstance(r, dict):
                    results.append({
                        "title": r.get('title', ''),
                        "link": r.get('href', ''),
                        "snippet": r.get('body', '')
                    })

    except Exception as e:
        print(f"⚠️ Lỗi nghiêm trọng khi search '{query}': {e}")
        # Không raise lỗi, chỉ in ra console server để debug

    return results