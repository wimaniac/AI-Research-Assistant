import concurrent.futures
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def scrape_url(url: str):
    """
    Hàm phụ trợ: Load 1 url duy nhất
    """
    try:
        loader = WebBaseLoader(
            url,
            header_template={
                "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
        docs = loader.load()
        return docs[0] if docs else None
    except Exception as e:
        print(f"Lỗi khi đọc đường dẫn {url}: {e}")

def scrape_urls(urls: list[str]):
    """
    Load danh sách url song song
    Nhanh hơn việc load tuần tự
    """
    print(f"Đang đọc nội dung từ {len(urls)} trang web....")
    res = []

    # Dùng ThreadPoolExecutor để chạy song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Gửi tasks vào pool
        future_to_url = {executor.submit(scrape_url,url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            doc = future.result()
            if doc and len(doc.page_content) > 500: # Chỉ lấy các bài viết đủ dài (> 500 ký tự)
                res.append(doc)
    print(f"Đã thu thập thành công {len(res)} tài liệu.")
    return res