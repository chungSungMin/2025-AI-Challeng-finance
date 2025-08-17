import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os
from datasets import load_dataset
from langchain_community.document_loaders import PyPDFLoader

# RAG를 위한 라이브러리 추가
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. 설정 (Configuration) ---

# 기본 모델 ID (학습에 사용한 모델)
BASE_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"

# ⭐️ 사용자 설정: 학습된 LoRA 어댑터가 저장된 경로를 지정해주세요.
LORA_ADAPTER_PATH = "/workspace/2025-AI-Challeng-finance/midm-lora-adapter-trainer/checkpoint-110" 

# 테스트 데이터 및 제출 파일 경로/workspace/paper_generate_mid_정보보호산업.jsonl
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_rag_inference.csv' 

# RAG 지식 베이스로 사용할 JSONL 파일 목록
RAG_DATA_FILES = [
"/workspace/개인금융채권의 관리 및 개인금융채무자의 보호에 관한 법률(법률)(제20369호)(20241017).pdf",
"/workspace/개인정보 보호법(법률)(제19234호)(20250313) (1).pdf",
"/workspace/거버넌스.pdf",
"/workspace/경찰공무원 등의 개인정보 처리에 관한 규정(대통령령)(제35039호)(20241203).pdf",
"/workspace/금융보안연구원.pdf",
"/workspace/금융소비자 보호에 관한 법률(법률)(제20305호)(20240814).pdf",
"/workspace/금융실명거래 및 비밀보장에 관한 법률 시행규칙(총리령)(제01406호)(20170726).pdf",
"/workspace/랜섬웨어.pdf",
"/workspace/마이데이터.pdf",
"/workspace/메타버스.pdf",
"/workspace/법원 개인정보 보호에 관한 규칙(대법원규칙)(제03109호)(20240315).pdf",
"/workspace/아웃소싱.pdf",
"/workspace/정보_보안.pdf",
"/workspace/클라우드컴퓨팅 발전 및 이용자 보호에 관한 법률(법률)(제20732호)(20250131).pdf",
]
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# 임베딩 모델 및 벡터 DB 경로 설정
# EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"
FAISS_DB_PATH = "./faiss_index_laws"


# --- 2. RAG 백엔드 구축 함수 ---
# def build_or_load_rag_backend():
#     """3개의 JSONL 파일로부터 FAISS 벡터 DB를 구축하거나 기존 DB를 로드합니다."""
    
#     # 1. 임베딩 모델 로드
#     embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
#     # 2. 이미 생성된 DB가 있으면 로드, 없으면 새로 생성
#     if os.path.exists(FAISS_DB_PATH):
#         print(f"⏳ 기존 벡터 DB를 '{FAISS_DB_PATH}'에서 로드합니다...")
#         vector_db = FAISS.load_local(
#             FAISS_DB_PATH, 
#             embedding_model, 
#             allow_dangerous_deserialization=True
#         )
#         print("✅ 벡터 DB 로드 완료.")
#     else:
#         print(f"⏳ '{RAG_DATA_FILES}' 파일들로 새로운 벡터 DB를 구축합니다...")
#         # 데이터셋 로드
#         dataset = load_dataset('json', data_files=RAG_DATA_FILES, split='train')
        
#         # Langchain Document 객체로 변환
#         # source_chunk를 검색 대상 텍스트로 사용
#         documents = [
#             Document(page_content=item['source_chunk'], metadata={'question': item['question'], 'answer': item['answer']}) 
#             for item in tqdm(dataset, desc="📄 Document 객체 생성 중")
#         ]
        
#         # FAISS 벡터 DB 생성
#         vector_db = FAISS.from_documents(documents, embedding_model)
        
#         # 다음 실행을 위해 로컬에 저장
#         vector_db.save_local(FAISS_DB_PATH)
#         print(f"✅ 새로운 벡터 DB 구축 및 저장 완료: '{FAISS_DB_PATH}'")

#     # 검색기(Retriever) 반환 (상위 3개 문서 검색)
#     return vector_db.as_retriever(search_kwargs={'k': 3})


# 필요한 PDF 로더를 import 합니다.
# --- 2. RAG 백엔드 구축 함수 (수정된 버전) ---
def build_or_load_rag_backend():
    """PDF 및 JSONL 파일들로부터 FAISS 벡터 DB를 구축하거나 기존 DB를 로드합니다."""
    
    # 1. 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 2. 이미 생성된 DB가 있으면 로드, 없으면 새로 생성
    if os.path.exists(FAISS_DB_PATH):
        print(f"⏳ 기존 벡터 DB를 '{FAISS_DB_PATH}'에서 로드합니다...")
        vector_db = FAISS.load_local(
            FAISS_DB_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ 벡터 DB 로드 완료.")
    else:
        print(f"⏳ '{RAG_DATA_FILES}' 파일들로 새로운 벡터 DB를 구축합니다...")
        
        # --- ★★★ 핵심 수정 부분 시작 ★★★ ---
        all_documents = []
        # 각 PDF 파일을 순회하며 텍스트를 추출합니다.
        for file_path in tqdm(RAG_DATA_FILES, desc="📚 PDF 파일 로딩 중"):
            try:
                # PyPDFLoader를 사용해 PDF 파일 로드
                loader = PyPDFLoader(file_path)
                # PDF의 각 페이지가 별도의 Document 객체로 분리되어 리스트로 반환됨
                documents_from_pdf = loader.load() 
                all_documents.extend(documents_from_pdf)
            except Exception as e:
                print(f"⚠️ 경고: '{file_path}' 파일을 처리하는 중 오류 발생: {e}")
        
        # --- ★★★ 핵심 수정 부분 끝 ★★★ ---

        if not all_documents:
            print("❌ 오류: 처리할 문서가 없습니다. RAG_DATA_FILES 경로를 확인해주세요.")
            exit()

        print(f"총 {len(all_documents)}개의 페이지(Document)를 임베딩합니다...")
        # FAISS 벡터 DB 생성 (기존 documents 변수명을 all_documents로 변경)
        vector_db = FAISS.from_documents(all_documents, embedding_model)
        
        # 다음 실행을 위해 로컬에 저장
        vector_db.save_local(FAISS_DB_PATH)
        print(f"✅ 새로운 벡터 DB 구축 및 저장 완료: '{FAISS_DB_PATH}'")

    # 검색기(Retriever) 반환 (상위 3개 문서 검색)
    return vector_db.as_retriever(search_kwargs={'k': 3})



# --- 3. 유틸리티 및 프롬프트 함수 ---

def is_multiple_choice(question_text: str) -> bool:
    """질문이 객관식인지 주관식인지 판별합니다."""
    lines = question_text.strip().split("\n")
    # 숫자로 시작하고 점(.) 또는 공백으로 끝나는 선택지가 2개 이상이면 객관식으로 판단
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?[\.\s]", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> tuple[str, list[str]]:
    """객관식 질문에서 순수 질문과 선택지 리스트를 분리합니다."""
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?[\.\s]", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options


# <<< 수정 1: RAG 검색 실패 시 사용할 Fallback 프롬프트 (첫 번째 코드와 동일하게 수정) >>>
def make_prompt(text: str) -> str:
    """
    RAG 없이 모델의 내부 지식만으로 답변을 생성하는 프롬프트입니다.
    (첫 번째 성공적인 코드의 프롬프트 구조와 동일)
    """
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        # 객관식 프롬프트
        prompt = f"""### 지시:
다음 질문에 대한 올바른 답변의 '번호'만 출력하세요. 다른 설명은 절대 추가하지 마세요.

### 질문:
{question}

### 선택지:
{chr(10).join(options)}

### 답변:
"""
    else:
        # 주관식 프롬프트
        prompt = f"""### 지시:
다음 질문에 대해 핵심 키워드를 중심으로 완벽한 한국어 문장으로 서술하세요.
'참고 문서'에 직접적인 언급이 없어도 최대한 배경 지식을 활용해서 답해주세요.
"문서에 따르면~ " 이라는 내용을 쓰지 말아주세요.


### 질문:
{text}

### 답변:
"""
    return prompt


# <<< 수정 2: RAG 전용 프롬프트 (첫 번째 코드의 성공적인 구조 기반으로 수정) >>>
def make_rag_prompt(text: str, context: str) -> str:
    """RAG 검색 결과를 포함하여 프롬프트를 생성합니다."""
    # 참고 문서가 비어있으면 RAG 없이 일반 프롬프트로 대체
    if not context.strip():
        return make_prompt(text)

    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        # 객관식 RAG 프롬프트
        prompt = f"""### 지시:
주어진 '참고 문서'를 바탕으로 다음 질문에 대한 올바른 답변의 '번호'만 출력하세요. 다른 설명은 절대 추가하지 마세요.

### 참고 문서:
{context}

### 질문:
{question}

### 선택지:
{chr(10).join(options)}

### 답변:
"""
    else:
        # 주관식 RAG 프롬프트
        prompt = f"""### 지시:
주어진 '참고 문서'의 내용을 바탕으로 다음 질문에 대해 완벽한 한국어 문장으로 서술하세요. '참고 문서에 따르면'과 같은 표현은 사용하지 마세요.
'참고 문서'에 직접적인 언급이 없어도 최대한 배경 지식을 활용해서 답해주세요.
"문서에 따르면~ " 이라는 내용을 쓰지 말아주세요.

### 참고 문서:
{context}

### 질문:
{text}

### 답변:
"""
    return prompt

# --- 4. 모델 및 토크나이저 로드 ---

print("⏳ 모델과 토크나이저를 로딩합니다...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"⏳ '{LORA_ADAPTER_PATH}'에서 LoRA 어댑터를 로딩하여 모델에 적용합니다...")
try:
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
except Exception as e:
    print(f"❌ 오류: LoRA 어댑터 로딩에 실패했습니다. 경로를 확인해주세요: {LORA_ADAPTER_PATH}")
    print(e)
    exit()

print("⏳ LoRA 가중치를 기본 모델에 병합합니다...")
model = model.merge_and_unload()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("✅ 모델 로딩 및 설정이 완료되었습니다.")


def post_process_answer(generated_text: str, original_question: str) -> str:
    """[수정] 생성된 텍스트에서 최종 답변을 추출하고 정리하는 강화된 함수입니다."""
    answer = generated_text.strip()
    
    if not answer:
        return "1"  # 답변이 아예 없는 경우 기본값 '1' 반환

    # 답변에 프롬프트 키워드가 포함된 경우 제거
    if "###" in answer:
        answer = answer.split("###")[0].strip()
        
    if is_multiple_choice(original_question):
        # 1단계: "정답은 5", "답은 5번"과 같이 명확한 패턴에서 숫자 추출
        match = re.search(r'(?:정답은|답은|선택은)\s*\D*(\d+)', answer)
        if match:
            return match.group(1)

        # 2단계: "5번", "5." 와 같은 패턴에서 숫자 추출
        match = re.search(r'\b(\d+)\s*(?:번|번입니다|\.)', answer)
        if match:
            return match.group(1)

        # 3단계: 문장 맨 앞에 있는 숫자 추출 (기존 로직)
        match = re.search(r"^\s*(\d+)", answer)
        if match:
            return match.group(1)

        # 4단계: 위 모든 조건에 해당하지 않을 경우, 텍스트 전체에서 처음 발견되는 숫자 추출
        match = re.search(r'(\d+)', answer)
        if match:
            return match.group(1)
            
        # 5단계: 그래도 숫자를 찾지 못하면 기본값 '1' 반환
        return "1"
    
    # 주관식 답변 처리
    return answer if answer else "답변을 생성하지 못했습니다."

# is_code_detected 함수는 그대로 유지
def is_code_detected(text: str) -> bool:
    """간단한 키워드 기반으로 생성된 텍스트에 코드가 포함되었는지 확인합니다."""
    code_keywords = ['def ', 'import ', 'class ', 'r\'', 'sys.stdout', 'ans_qna']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in code_keywords)


def is_code_detected(text: str) -> bool:
    """간단한 키워드 기반으로 생성된 텍스트에 코드가 포함되었는지 확인합니다."""
    code_keywords = ['def ', 'import ', 'class ', 'r\'', 'sys.stdout', 'ans_qna']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in code_keywords)


# --- 6. 메인 실행 (RAG 적용) ---
if __name__ == "__main__":
    # RAG 백엔드(Retriever) 준비
    retriever = build_or_load_rag_backend()

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    preds = []
    MAX_RETRIES = 3 

    for index, q in tqdm(enumerate(test_df['Question']), total=len(test_df), desc="🚀 RAG 추론 진행 중"):
        
        # RAG 검색 단계 추가
        try:
            # 질문(q)을 기반으로 관련 문서를 검색합니다.
            retrieved_docs = retriever.get_relevant_documents(q)
            # 검색된 문서들의 내용을 하나의 문자열로 합칩니다.
            context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"⚠️ TEST_{index} 질문 검색 중 오류 발생: {e}")
            context_text = "" # 검색 실패 시 context는 비워둠

        # RAG 프롬프트 생성
        prompt = make_rag_prompt(q, context_text)
        
        is_valid_answer = False
        retries = 0
        generated_text = ""

        while not is_valid_answer and retries < MAX_RETRIES:
            if retries > 0:
                print(f"\n🔄 TEST_{index} 질문에 대한 답변 재시도 중... ({retries}/{MAX_RETRIES})")

            output = pipe(
                prompt, 
                max_new_tokens=512,
                temperature=0.1 + (retries * 0.15),
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = output[0]['generated_text']

            if is_code_detected(generated_text):
                retries += 1
                if retries == MAX_RETRIES:
                    print(f"❌ TEST_{index} 질문에 대해 최대 재시도 횟수 초과. 마지막으로 생성된 답변을 사용합니다.")
                    is_valid_answer = True
            else:
                is_valid_answer = True

        pred_answer = post_process_answer(generated_text, original_question=q)
        preds.append(pred_answer)

    print("\n📄 추론이 완료되었습니다. 제출 파일을 생성합니다...")
    try:
        sample_submission = pd.read_csv('/workspace/open/sample_submission.csv')
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 제출 파일 생성이 완료되었습니다: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print(f"❌ 오류: '/workspace/open/sample_submission.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
