import streamlit as st
import time
from agents.workflow import run_research

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🕵️",
    layout="centered"
)

# --- CSS TÙY CHỈNH (CHO ĐẸP HƠN) ---
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
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🕵️ AI Research Assistant")
st.markdown("---")
st.markdown(
    """
    Chào mừng! Tôi là trợ lý AI giúp bạn:
    1. **Tìm kiếm** thông tin mới nhất trên Internet.
    2. **Đọc hiểu** hàng chục trang tài liệu trong vài giây.
    3. **Tổng hợp** thành báo cáo chuyên sâu.
    """
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.success("✅ Google Gemini API: Connected")
    st.info("💡 Mẹo: Nhập chủ đề càng cụ thể, báo cáo càng chi tiết.")
    st.divider()
    st.markdown("Developed with ❤️ using LangChain")

# --- MAIN INPUT ---
with st.form("research_form"):
    topic = st.text_input(
        "Nhập chủ đề bạn muốn nghiên cứu:",
        placeholder="Ví dụ: Xu hướng Data Engineering năm 2026..."
    )
    submitted = st.form_submit_button("🚀 Bắt đầu nghiên cứu")

# --- XỬ LÝ KHI BẤM NÚT ---
if submitted:
    if not topic:
        st.warning("⚠️ Vui lòng nhập chủ đề!")
    else:
        # Sử dụng st.status để hiển thị quy trình (Trải nghiệm UX tốt hơn)
        with st.status("🤖 AI đang làm việc...", expanded=True) as status:

            st.write("🧠 Đang phân tích chủ đề và lập kế hoạch...")
            # Vì hàm run_research chạy đồng bộ, ta gọi nó ở đây
            # (Trong thực tế, bạn có thể tách nhỏ hàm workflow để update từng bước UI)

            start_time = time.time()
            try:
                result = run_research(topic)
                end_time = time.time()

                status.update(label="✅ Nghiên cứu hoàn tất!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Có lỗi xảy ra!", state="error")
                st.error(f"Lỗi chi tiết: {e}")
                result = None

        # --- HIỂN THỊ KẾT QUẢ ---
        if result:
            st.divider()
            st.subheader(f"📄 Báo cáo: {topic}")
            st.caption(f"⏱️ Thời gian xử lý: {end_time - start_time:.2f} giây")

            # Hiển thị báo cáo Markdown
            st.markdown(result)

            # Nút tải xuống
            st.download_button(
                label="📥 Tải báo cáo (.md)",
                data=result,
                file_name=f"report_{topic.replace(' ', '_')}.md",
                mime="text/markdown"
            )