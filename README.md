# 🕵️ AI Research Assistant (End-to-End Agentic RAG)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Ollama Local](https://img.shields.io/badge/AI-Ollama_Local-orange)

**AI Research Assistant** là một hệ thống **Agentic RAG có trạng thái (Stateful)** được xây dựng trên nền tảng LangGraph. Thay vì chỉ tìm kiếm tuyến tính, hệ thống hoạt động như một nhóm chuyên gia: tự động lập kế hoạch, thu thập dữ liệu, tự đánh giá tính đầy đủ của thông tin, lặp lại quá trình tìm kiếm nếu cần, và cuối cùng tổng hợp thành một báo cáo chuyên sâu.

Điểm đặc biệt của dự án là việc triển khai kỹ thuật **Hybrid Search (Tìm kiếm lai)** kết hợp giữa `FAISS` (Vector Search) và `BM25` (Keyword Search) để tối ưu hóa độ chính xác khi truy xuất thông tin.

## Tính năng nổi bật

* **Luồng đa tác nhân (Multi-Agent Workflow):** Quản lý quy trình phức tạp bằng đồ thị trạng thái LangGraph (Plan -> Retrieve -> Evaluate -> Refine -> Summarize).
* **Lập kế hoạch tự động (Auto-Planning):** AI tự động phân tích chủ đề và sinh ra các từ khóa tìm kiếm tối ưu nhất.
* **Thu thập dữ liệu thời gian thực:** Sử dụng Tavily và Web Scraper đa luồng (Multi-threading) để đọc nội dung từ Internet.
* **Vòng lặp tự đánh giá (Self-Reflection):** Tác nhân "Evaluator" sẽ kiểm tra xem dữ liệu thu thập đã đủ chưa, nếu chưa sẽ yêu cầu tìm kiếm thêm.
* **Hybrid Search (RAG nâng cao):**
    * Sử dụng **FAISS** để tìm kiếm theo ngữ nghĩa (Semantic Search).
    * Sử dụng **BM25** để tìm kiếm theo từ khóa chính xác (Lexical Search).
    * Kết hợp bằng thuật toán **Ensemble Retriever** (Custom Implementation).
    * Tích hợp **Cross-Encoder Reranker** để xếp hạng lại tài liệu, tối ưu hóa Context Window.
* **Trích dẫn nguồn rõ ràng:** Báo cáo đầu ra luôn đính kèm danh sách các URL đã tham khảo.
* **Giao diện thời gian thực:** Streamlit UI hiển thị trực tiếp tiến trình suy nghĩ và làm việc của AI (Streaming Progress).
* **Kiến trúc Tách biệt (Decoupled Architecture):** Backend xử lý AI bằng FastAPI, Frontend hiển thị bằng Streamlit, giao tiếp qua REST API (NDJSON streaming).

## Công nghệ sử dụng

* **Language:** Python
* **Framework:** LangChain & LangGraph
* **LLM & Embeddings:** Llama 3.1 (via Ollama) & BGE Embeddings (via HuggingFace)
* **Vector Store:** FAISS (CPU)
* **Retrieval Algorithm:** BM25 + Ensemble Retriever + Cross-Encoder
* **Backend:** FastAPI & Uvicorn
* **Frontend:** Streamlit (via Requests)
* **Search Engine:** Tavily Search API

## Cài đặt và Chạy dự án

### 1. Yêu cầu
- Python 3.10+
- [Ollama](https://ollama.com/) đã được cài đặt và đang chạy. Tải mô hình `llama3.1`:
  ```bash
  ollama pull llama3.1
  ```
- API Key cho Tavily Search.

### 2. Clone Repository
```bash
git clone https://github.com/wimaniac/AI-Research-Assistant
cd AI-Research-Assistant
```
### 2. Thiết lập môi trường ảo
```bash
python -m venv venv
.\venv\Scripts\activate
```
### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
### 4. Cấu hình API Key
```bash
GOOGLE_API_KEY=YOUR_API_KEY
```
### 5. Chạy ứng dụng
```bash
streamlit run main.py
```
