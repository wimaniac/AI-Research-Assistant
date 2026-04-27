from langchain_core.prompts import ChatPromptTemplate

# 1. Prompt cho Planner (Lập kế hoạch tìm kiếm)
# Nhiệm vụ: Tách chủ đề lớn thành N từ khóa/câu hỏi tìm kiếm tối ưu
PLANNER_PROMPT = ChatPromptTemplate.from_template("""
Bạn là người lập kế hoạch nghiên cứu.

Cho một chủ đề của người dùng, hãy chia nhỏ nó thành nhiều câu hỏi phụ cụ thể mà khi kết hợp lại sẽ bao quát đầy đủ chủ đề đó.

Yêu cầu:
- Mỗi câu hỏi phải rõ ràng và tập trung
- Tránh trùng lặp
- Bao quát các khía cạnh khác nhau của chủ đề

Chủ đề: {user_query}

QUAN TRỌNG: CHỈ trả về duy nhất một mảng JSON chứa các câu hỏi. KHÔNG thêm bất kỳ từ ngữ giao tiếp, lời chào, hay giải thích nào (như "Dưới đây là...", "Hy vọng...").
Ví dụ đúng:
[
    "câu hỏi 1", 
    "câu hỏi 2"
]
""")

# 2. Prompt cho Evaluator (Đánh giá thông tin)
EVALUATOR_PROMPT = ChatPromptTemplate.from_template("""
Bạn là một chuyên gia đánh giá thông tin.
Chủ đề nghiên cứu gốc: "{topic}"

Dưới đây là các thông tin đã thu thập được cho đến hiện tại:
--- THÔNG TIN ĐÃ THU THẬP ---
{context}
-----------------------------

Nhiệm vụ:
Đánh giá xem thông tin đã thu thập có đủ để viết một báo cáo toàn diện và chi tiết về chủ đề gốc hay chưa.
- Nếu ĐÃ ĐỦ: Trả về chính xác chữ "NO".
- Nếu CÒN THIẾU (cần tìm thêm thông tin về khía cạnh nào đó): Trả về chữ "YES" ở dòng đầu tiên, và ở các dòng tiếp theo liệt kê các câu truy vấn (search queries) mới để tìm kiếm thêm thông tin bị thiếu (mỗi dòng 1 câu truy vấn).

Lưu ý: Chỉ trả về định dạng được yêu cầu, không giải thích dài dòng, không sinh ra câu từ thừa.
Ví dụ thiếu:
YES
Query mới 1
""")

# 3. Prompt cho Synthesizer / Writer (Viết báo cáo)
# Nhiệm vụ: Dùng thông tin đã tìm được (Context) để trả lời người dùng
WRITER_PROMPT = ChatPromptTemplate.from_template("""
Bạn là một trợ lý viết báo cáo chuyên nghiệp.
Dưới đây là các thông tin thu thập được từ internet về chủ đề: "{topic}"

--- DỮ LIỆU THU THẬP (CONTEXT) ---
{context}
-----------------------------------

Yêu cầu:
1. Viết một báo cáo chi tiết, có cấu trúc rõ ràng (Markdown) dựa trên dữ liệu trên.
2. Báo cáo cần có: Tóm tắt, Các điểm chính, Chi tiết chuyên sâu, và Kết luận.
3. Giọng văn khách quan, chuyên nghiệp.
4. Nếu thông tin trong Context không đủ, hãy nói rõ là thiếu thông tin, đừng bịa đặt (hallucination).
5. Trình bày đẹp mắt với các tiêu đề (##), gạch đầu dòng...
6. Trích dẫn nguồn trong bài và LUÔN LUÔN thêm một phần "## 🔗 Nguồn tham khảo" ở cuối báo cáo. Các đường link phải được định dạng Markdown để có thể click được (ví dụ: `<URL>` hoặc `Link`).

Bắt đầu viết báo cáo:
""")