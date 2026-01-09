# 🕵️ AI Research Assistant (End-to-End RAG)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Google Gemini](https://img.shields.io/badge/AI-Google_Gemini-orange)

**AI Research Assistant** là một hệ thống Agentic RAG thông minh giúp tự động hóa quy trình nghiên cứu thông tin. Thay vì chỉ tìm kiếm từ khóa đơn giản, hệ thống lập kế hoạch, thu thập dữ liệu từ Internet, đọc hiểu hàng chục trang tài liệu và tổng hợp thành báo cáo chuyên sâu.

Điểm đặc biệt của dự án là việc triển khai kỹ thuật **Hybrid Search (Tìm kiếm lai)** kết hợp giữa `FAISS` (Vector Search) và `BM25` (Keyword Search) để tối ưu hóa độ chính xác khi truy xuất thông tin.

## Tính năng nổi bật

* **Lập kế hoạch tự động (Auto-Planning):** AI tự động phân tích chủ đề và sinh ra các từ khóa tìm kiếm tối ưu nhất.
* **Thu thập dữ liệu thời gian thực:** Sử dụng DuckDuckGo Search và Web Scraper đa luồng (Multi-threading) để đọc nội dung từ Internet.
* **Hybrid Search (RAG nâng cao):**
    * Sử dụng **FAISS** để tìm kiếm theo ngữ nghĩa (Semantic Search).
    * Sử dụng **BM25** để tìm kiếm theo từ khóa chính xác (Lexical Search).
    * Kết hợp bằng thuật toán **Ensemble Retriever** (Custom Implementation).
* **Viết báo cáo tự động:** Tổng hợp thông tin từ nhiều nguồn và viết báo cáo Markdown có cấu trúc.
* **Tốc độ cao:** Sử dụng Google Gemini 1.5 Flash cho tốc độ xử lý Context Window lớn cực nhanh.

## Công nghệ sử dụng

* **Language:** Python
* **Framework:** [LangChain](https://www.langchain.com/)
* **LLM & Embeddings:** Google Gemini (via `langchain-google-genai`)
* **Vector Store:** FAISS (CPU)
* **Retrieval Algorithm:** BM25 + Ensemble Retriever
* **Frontend:** Streamlit
* **Search Engine:** DuckDuckGo

## Cài đặt và Chạy dự án

### 1. Clone Repository
```bash
git clone https://github.com/wimaniac/AI-Research-Assistant
cd ai-research-assistant
### 2. Thiết lập môi trường ảo 
python -m venv venv
.\venv\Scripts\activate
### 3. Cài đặt thư viện
pip install -r requirements.txt
### 4. Cấu hình API Key
GOOGLE_API_KEY=YOUR_API_KEY
### 5. Chạy ứng dụng
streamlit run main.py
