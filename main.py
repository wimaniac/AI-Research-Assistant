import os
import warnings
import logging
import sys
import requests
import json

# --- CẤU HÌNH LOGGING TRUNG TÂM ---
# Cấu hình này sẽ áp dụng cho toàn bộ ứng dụng, giúp quan sát (Observability) tốt hơn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout, # In log ra console
)

# Giảm mức độ log của các thư viện bên thứ ba để tránh làm nhiễu terminal
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
import time

# API URL Của FastAPI Backend
API_URL = "http://localhost:8000"

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🕵️",
    layout="centered"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .report-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- KHỞI TẠO SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_id" not in st.session_state:
    st.session_state.request_id = None
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    if st.button("🗑️ Xóa lịch sử & Làm mới"):
        st.session_state.messages = []
        st.session_state.request_id = None
        st.session_state.last_report = None
        st.session_state.last_topic = None
        try:
            requests.post(f"{API_URL}/clear_cache")
        except Exception:
            pass
        st.rerun()

    st.info("💡 Mẹo: Sau khi có báo cáo, bạn có thể chat hỏi thêm chi tiết bên dưới.")
    st.divider()
# --- HEADER ---
st.title("🕵️ AI Research Assistant")
st.markdown("---")

# --- PHẦN 1: FORM NHẬP CHỦ ĐỀ ---
with st.form("research_form"):
    topic = st.text_input(
        "Nhập chủ đề bạn muốn nghiên cứu:",
        placeholder="Ví dụ: Xu hướng Data Engineering năm 2026..."
    )
    submitted = st.form_submit_button("🚀 Bắt đầu nghiên cứu")

# --- XỬ LÝ KHI BẤM NÚT NGHIÊN CỨU ---
if submitted:
    if not topic:
        st.warning("⚠️ Vui lòng nhập chủ đề!")
    elif topic.strip().lower() == str(st.session_state.last_topic).strip().lower():
        st.info("💡 Bạn đã nghiên cứu chủ đề này rồi! Kết quả đang hiển thị bên dưới. Nếu muốn nghiên cứu lại từ đầu, hãy ấn nút 'Xóa lịch sử & Làm mới' bên trái.")
    else:
        # Reset lại trạng thái cũ
        st.session_state.messages = []
        st.session_state.request_id = None
        st.session_state.last_report = None

        with st.status("🤖 AI đang làm việc...", expanded=True) as status:
            st.write("🚀 Gửi yêu cầu lên hệ thống Backend...")
            
            try:
                # Gọi API dạng Stream
                response = requests.post(f"{API_URL}/research", json={"topic": topic}, stream=True)
                
                if response.status_code == 429:
                    st.error("⏳ Hệ thống đang bận xử lý một yêu cầu khác. Vui lòng chờ vài phút và thử lại! (Concurrency Limit: 1)")
                    status.update(label="Hệ thống bận", state="error")
                else:
                    result_pack = None
                    # Hứng từng dòng NDJSON trả về
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if data["type"] == "progress":
                                st.write(data["msg"])
                            elif data["type"] == "result":
                                result_pack = data

                    if result_pack:
                        st.session_state.last_report = result_pack["report"]
                        st.session_state.request_id = result_pack["request_id"]
                        st.session_state.last_topic = topic
                        latency_str = f"{result_pack['latency']:.2f}s"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**Báo cáo nghiên cứu về: {topic}**\n*(ReqID: {st.session_state.request_id} | Latency: {latency_str})*\n\n" + result_pack["report"]
                        })
                        status.update(label="Nghiên cứu hoàn tất!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Không thể kết nối đến máy chủ Backend!", state="error")
                st.error(f"Lỗi chi tiết: {e}")

# --- PHẦN 2: HIỂN THỊ KẾT QUẢ & CHAT ---

# Nếu đã có báo cáo thì mới hiện khu vực này
if st.session_state.last_report:
    st.divider()

    # Hiển thị lịch sử chat (Bao gồm cả báo cáo ban đầu và các câu hỏi sau đó)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Nếu là tin nhắn báo cáo đầu tiên, hiện thêm nút tải
            if msg["content"].startswith("**Báo cáo nghiên cứu về:"):
                st.download_button(
                    label="📥 Tải báo cáo (.md)",
                    data=st.session_state.last_report,
                    file_name="research_report.md",
                    mime="text/markdown"
                )

    # --- PHẦN 3: XỬ LÝ CHAT INPUT ---
    if st.session_state.request_id:
        if user_input := st.chat_input("Hỏi thêm chi tiết về báo cáo này..."):
            # 1. Hiện câu hỏi của User
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 2. Giao tiếp với API để AI trả lời
            with st.chat_message("assistant"):
                with st.spinner("Đang đọc lại tài liệu..."):
                    try:
                        res = requests.post(f"{API_URL}/chat", json={
                            "request_id": st.session_state.request_id,
                            "question": user_input
                        })
                        answer = res.json().get("answer", "Lỗi không xác định từ Backend")
                    except Exception:
                        answer = "⚠️ Lỗi: Không thể kết nối tới Backend để chat."
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})