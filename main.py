import streamlit as st
import time
from agents.workflow import run_research
from core.llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "last_report" not in st.session_state:
    st.session_state.last_report = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    if st.button("🗑️ Xóa lịch sử & Làm mới"):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.last_report = None
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
    else:
        # Reset lại trạng thái cũ
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.last_report = None

        with st.status("🤖 AI đang làm việc...", expanded=True) as status:
            st.write("🧠 Đang lập kế hoạch & Tìm kiếm dữ liệu...")
            start_time = time.time()

            try:
                # Gọi hàm research (Lưu ý: hàm này giờ trả về Dict)
                result_pack = run_research(topic)
                end_time = time.time()

                # Lưu kết quả vào Session State
                st.session_state.last_report = result_pack["report"]
                st.session_state.retriever = result_pack["retriever"]

                # Thêm báo cáo vào lịch sử chat như tin nhắn đầu tiên của AI
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Báo cáo nghiên cứu về: {topic}**\n\n" + result_pack["report"]
                })

                status.update(label="Nghiên cứu hoàn tất!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ Có lỗi xảy ra!", state="error")
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
    # Chỉ hiện ô chat khi đã có retriever (đã nghiên cứu xong)
    if st.session_state.retriever:
        if user_input := st.chat_input("Hỏi thêm chi tiết về báo cáo này..."):
            # 1. Hiện câu hỏi của User
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 2. AI trả lời (Sử dụng Context từ Retriever)
            with st.chat_message("assistant"):
                with st.spinner("Đang đọc lại tài liệu..."):
                    llm = get_llm()
                    retriever = st.session_state.retriever

                    # Tìm kiếm thông tin liên quan đến câu hỏi user trong bộ nhớ cũ
                    related_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join([d.page_content for d in related_docs])

                    # Prompt chuyên biệt cho Chat
                    chat_prompt = ChatPromptTemplate.from_template("""
                    Bạn là trợ lý nghiên cứu. Người dùng đang hỏi về báo cáo đã tạo.

                    Dữ liệu liên quan tìm thấy (Context):
                    {context}

                    Câu hỏi của người dùng: {question}

                    Hãy trả lời ngắn gọn, súc tích dựa trên Context trên. 
                    Nếu không có thông tin trong Context, hãy nói là "Dữ liệu thu thập được chưa đề cập đến vấn đề này".
                    """)

                    # Chạy Chain
                    chain = chat_prompt | llm | StrOutputParser()
                    response = chain.invoke({"context": context_text, "question": user_input})

                    # Hiện câu trả lời & Lưu vào history
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})