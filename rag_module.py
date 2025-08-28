from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


vector_store = None

def load_pdf_to_vectorstore(pdf_path: str):
    global vector_store
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"[RAG] PDF {pdf_path} 로드 및 벡터DB 구축 완료.")
    return vector_store  

def retrieve_from_pdf(query: str, top_k: int = 3):
    global vector_store
    if vector_store is None:
        return "[RAG] PDF가 로드되지 않았습니다."
    results = vector_store.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in results])

