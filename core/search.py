from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
def search_web(query: str, max_results: int = 5):
    """
    Tìm kếm trên DuckDuckGo và trả về danh sách các URL và tiêu đề.
    Args:
        query: câu truy vấn (vd: tương lai ngành Data Engineering)
        max_results: Số lượng kết quả muốn lấy
    Returns:
      List[dict]: danh sách chứa các "title", "link", "snippet"
    """
    print(f"Đang tìm kiếm {query}....")
    #Khởi tạo wrapper của DuckDuckGo
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)

    #Lấy kết quả trả về
    results = wrapper.results(query=query, max_results=max_results)
    return results

if __name__ == "__main__":
    res = search_web("LangChain tutorial python",3)
    for r in res:
        print(f"-[{r['title']}]-({r['link']})")