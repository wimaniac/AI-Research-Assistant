from langchain_core.prompts import ChatPromptTemplate

# 1. Prompt cho Planner (Lập kế hoạch tìm kiếm)
# Nhiệm vụ: Tách chủ đề lớn thành 3 từ khóa/câu hỏi tìm kiếm tối ưu cho Google/DuckDuckGo
PLANNER_PROMPT = ChatPromptTemplate.from_template("""
Bạn là một chuyên gia nghiên cứu thông tin (AI Research Assistant).
Nhiệm vụ của bạn là lập kế hoạch tìm kiếm thông tin cho chủ đề: "{topic}"

Hãy đưa ra danh sách 3 câu truy vấn (search queries) tối ưu nhất để tìm kiếm trên Google nhằm thu thập đủ thông tin toàn diện về chủ đề này.
Các câu truy vấn nên đa dạng (định nghĩa, xu hướng, số liệu, so sánh...).

Yêu cầu trả về: CHỈ TRẢ VỀ CÁC CÂU TRUY VẤN, ngăn cách nhau bằng dấu xuống dòng. Không giải thích thêm.
Ví dụ:
Query 1
Query 2
Query 3
""")

# 2. Prompt cho Writer (Viết báo cáo)
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

Bắt đầu viết báo cáo:
""")