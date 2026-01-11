from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List


class SimpleEnsembleRetriever(BaseRetriever):
    """Bộ tìm kiếm lai đơn giản kết hợp nhiều retrievers"""

    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Lấy tài liệu liên quan từ tất cả retrievers và merge"""
        doc_scores = {}

        # Lấy kết quả từ mỗi retriever
        for retriever, weight in zip(self.retrievers, self.weights):
            # Sử dụng invoke() thay vì get_relevant_documents()
            docs = retriever.invoke(query)
            for i, doc in enumerate(docs):
                # Tính điểm dựa trên vị trí (doc đầu có điểm cao hơn)
                score = weight * (len(docs) - i) / len(docs)
                doc_id = doc.page_content[:100]  # Dùng 100 ký tự đầu làm ID

                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += score
                else:
                    doc_scores[doc_id] = {'doc': doc, 'score': score}

        # Sắp xếp theo điểm và trả về
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs[:4]]  # Trả về top 4


def create_retriever(documents, embeddings):
    """
    Tạo ra 1 hệ thống tìm kiếm lai (Hybird Search) từ danh sách tài liệu.
    Kết hợp giữa vector search (FAISS) và keyword search(BM25)
    Args:
        documents: List[Document] - Danh sách tài liệu thô đã scrape
        embeddings: Model embeddings (google)
    Returns:
        SimpleEnsembleRetriever: Bộ tìm kiếm đã tối ưu
    """
    # 1 Chunking: Cắt nhỏ văn bản
    # cắt nhỏ để vừa vặn với Context Window của LLM và tăng độ chính xác khi tìm kiếm.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],  # cắt theo đoạn văn, rồi đến câu
    )
    splits = text_splitter.split_documents(documents)
    if not splits:
        print("Không có dữ liệu để cắt.")
        return None
    print(f'Đã chia thành {len(splits)} chunk nhỏ.')

    # Vector search(Sematic) - Dùng FAISS
    # Tìm kiếm dựa trên ý nghĩa (vd: "AI" tìm ra "Trí tuệ nhân tạo"
    vectorstores = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstores.as_retriever(search_kwargs={"k": 4})

    # Keyword search - dùng bm25
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 4

    # Hybrid search - Kết hợp cả hai
    # Với trọng số 0.5 giúp cân bằng  giữa ngữ nghĩa và bắt trúng từ khóa
    ensemble_retriever = SimpleEnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever