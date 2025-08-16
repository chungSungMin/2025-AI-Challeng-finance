# generate_RAG.py

import os
from tqdm import tqdm
# from datasets import load_dataset  # 더 이상 필요 없으므로 삭제하거나 주석 처리

# PDF 처리 및 텍스트 분할을 위한 라이브러리 추가
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 설정 (Configuration) ---
# 📄 이제 JSONL이 아닌 PDF 파일 경로를 지정합니다.
RAG_DATA_FILES = [
    "/workspace/2025-AI-Challeng-finance/개인금융채권의 관리 및 개인금융채무자의 보호에 관한 법률(법률)(제20369호)(20241017).pdf",
"/workspace/2025-AI-Challeng-finance/개인정보 보호법(법률)(제19234호)(20250313) (1).pdf",
"/workspace/2025-AI-Challeng-finance/거버넌스.pdf",
"/workspace/2025-AI-Challeng-finance/경찰공무원 등의 개인정보 처리에 관한 규정(대통령령)(제35039호)(20241203).pdf",
"/workspace/2025-AI-Challeng-finance/금융보안연구원.pdf",
"/workspace/2025-AI-Challeng-finance/금융소비자 보호에 관한 법률(법률)(제20305호)(20240814).pdf",
"/workspace/2025-AI-Challeng-finance/금융실명거래 및 비밀보장에 관한 법률 시행규칙(총리령)(제01406호)(20170726).pdf",
"/workspace/2025-AI-Challeng-finance/랜섬웨어.pdf",
"/workspace/2025-AI-Challeng-finance/마이데이터.pdf",
"/workspace/2025-AI-Challeng-finance/메타버스.pdf",
"/workspace/2025-AI-Challeng-finance/법원 개인정보 보호에 관한 규칙(대법원규칙)(제03109호)(20240315).pdf",
"/workspace/2025-AI-Challeng-finance/아웃소싱.pdf",
"/workspace/2025-AI-Challeng-finance/정보_보안.pdf",
"/workspace/2025-AI-Challeng-finance/클라우드컴퓨팅 발전 및 이용자 보호에 관한 법률(법률)(제20732호)(20250131).pdf",
]
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
FAISS_DB_PATH = "./faiss_index_laws"


# ⭐️ PDF를 읽고 텍스트로 분할하는 함수 (새로 추가)
def load_and_chunk_pdfs(file_paths):
    all_docs = []
    for file_path in file_paths:
        print(f"📄 '{os.path.basename(file_path)}' 파일 처리 중...")
        # 1. PDF 파일에서 텍스트 추출
        reader = PdfReader(file_path)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() if page.extract_text() else ""

        # 2. 추출된 텍스트를 의미 단위로 분할 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # 청크 크기
            chunk_overlap=100, # 청크 간 겹치는 부분
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_text(raw_text)

        # 3. 분할된 청크를 Document 객체로 변환
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={"source": os.path.basename(file_path)}))
    
    print(f"✅ 총 {len(all_docs)}개의 문서(chunk)를 생성했습니다.")
    return all_docs


# ⭐️ PDF 처리 로직을 적용한 DB 생성 함수 (수정)
def create_and_save_vector_db():
    """
    PDF 파일들을 읽고 처리하여 FAISS 벡터 DB를 생성하고 로컬에 저장합니다.
    """
    # 1. 임베딩 모델 로드
    print(f"🚀 임베딩 모델({EMBEDDING_MODEL_NAME})을 로드합니다...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료.")

    # 2. PDF 로드 및 처리 (load_dataset 대신 새로 만든 함수 사용)
    documents = load_and_chunk_pdfs(RAG_DATA_FILES)
    
    # 3. FAISS 벡터 DB 생성
    print("🧠 FAISS 벡터 DB를 생성합니다. 시간이 소요될 수 있습니다...")
    vector_db = FAISS.from_documents(documents, embedding_model)
    
    # 4. 생성된 DB를 로컬에 저장
    vector_db.save_local(FAISS_DB_PATH)
    print(f"✅ 새로운 벡터 DB 구축 및 저장 완료: '{FAISS_DB_PATH}'")


if __name__ == "__main__":
    if os.path.exists(FAISS_DB_PATH):
        print(f"⚠️ 경고: '{FAISS_DB_PATH}'에 이미 DB가 존재합니다.")
        user_input = input("기존 DB를 덮어쓰시겠습니까? (y/n): ").lower()
        if user_input == 'y':
            create_and_save_vector_db()
        else:
            print("작업을 취소했습니다.")
    else:
        create_and_save_vector_db()