import os
import torch
import pandas as pd
import re
from tqdm import tqdm
import json

# --- 1. 라이브러리 Import ---

# LangChain (RAG)
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Hugging Face (LLM)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# from peft import PeftModel # ⭐️ PeftModel은 더 이상 필요 없으므로 주석 처리하거나 삭제합니다.

# --- 2. 설정 (Configuration) ---

# 기본 모델 ID
BASE_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"

# ⭐️ 사용자 설정: 학습된 LoRA 어댑터 경로 (더 이상 사용하지 않음)
# LORA_ADAPTER_PATH = "/workspace/checkpoint-708" 

# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_rag_langchain_v1.csv' 

# RAG 지식 베이스로 사용할 JSONL 파일 경로
RAG_JSONL_PATH = "/workspace/2025-AI-Challeng-finance/cybersecurity_data_regex_cleaned.jsonl"

EMBEDDING_MODEL_NAME = "BM-K/KoSimCSE-roberta-multitask"
FAISS_DB_PATH = "./faiss_index_langchain_jsonl" 


# --- 3. RAG 백엔드 구축 함수 (LangChain 방식으로 수정) ---
def build_or_load_rag_backend_langchain():
    """LangChain을 사용하여 FAISS 벡터 DB를 구축하거나 기존 DB를 로드합니다."""
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if os.path.exists(FAISS_DB_PATH):
        print(f"⏳ 기존 벡터 DB를 '{FAISS_DB_PATH}'에서 로드합니다...")
        vector_db = FAISS.load_local(
            FAISS_DB_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ 벡터 DB 로드 완료.")
    else:
        print(f"⏳ '{RAG_JSONL_PATH}' 파일로 새로운 벡터 DB를 구축합니다...")
        
        if not os.path.exists(RAG_JSONL_PATH):
            print(f"❌ 치명적 오류: RAG 데이터 파일 '{RAG_JSONL_PATH}'을(를) 찾을 수 없습니다.")
            return None

        # JSONL 파일을 LangChain Document 객체로 로드
        loader = JSONLoader(
            file_path=RAG_JSONL_PATH,
            jq_schema='"질문: " + .question + "\\n답변 : " + .answer', 
            json_lines=True,
            text_content=False
        )
        all_documents = loader.load()

        if not all_documents:
            print("❌ 오류: 처리할 문서가 없습니다. JSONL 파일의 내용이나 경로를 확인해주세요.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(all_documents)
        print(f"✅ 문서를 총 {len(split_documents)}개의 작은 조각(chunk)으로 분할했습니다.")

        print(f"총 {len(split_documents)}개의 조각(chunk)을 임베딩합니다...")
        vector_db = FAISS.from_documents(split_documents, embedding_model)
        
        vector_db.save_local(FAISS_DB_PATH)
        print(f"✅ 새로운 벡터 DB 구축 및 저장 완료: '{FAISS_DB_PATH}'")

    # LangChain의 Retriever 객체 반환
    return vector_db.as_retriever(search_kwargs={'k': 3})


# --- 4. 유틸리티 및 프롬프트 함수 (기존 코드와 동일) ---

def is_multiple_choice(question_text: str) -> bool:
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?[\.\s]", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> tuple[str, list[str]]:
    lines = full_text.strip().split("\n")
    q_lines, options = [], []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?[\.\s]", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options


def make_rag_prompt(text: str, context: str) -> str:
    """RAG 검색 결과를 우선적으로 사용하여 프롬프트를 생성합니다."""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = f"""### 지시:
"Please reason step by step, and you should must write the correct option number (1, 2, 3, 4 or 5).\n 정답 번호를 반드시 하나만 출력하세요. 설명은 필요 없습니다."
정답을 도출할 때 참고자료에 관련 내용이나 단어가 있다면 답안 선택에 **반드시 활용**하세요

### 참고 문서:
{context}

### 질문:
{question}

### 선택지:
{chr(10).join(options)}

### 답변:
"""
    else:
        prompt = f"""### 지시:
주어진 **'참고 문서'의 내용만을 근거**로 하여 '질문'에 답변하세요.
문서에서 질문과 관련된 핵심 내용을 종합하여, 전문 용어를 사용해 2~3개의 완벽한 한국어 문장으로 설명해야 합니다.
**'참고 문서에 따르면'과 같은 표현은 절대 사용하지 마세요.** 당신의 배경 지식이나 외부 정보는 사용하지 마세요.
Generate your thought process step by step, but don't print it out.

### 참고 문서:
{context}

### 질문:
{text}

### 답변:
"""
    return prompt



# --- 5. 모델 및 토크나이저 로드 (⭐️ LoRA 어댑터 로드 제거) ---
print("⏳ 모델과 토크나이저를 로딩합니다...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ⭐️ 'base_model' 대신 'model' 변수명으로 바로 로드하여 파이프라인에 사용합니다.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    quantization_config=quantization_config, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ⭐️ LoRA 어댑터를 로드하고 병합하는 부분을 완전히 제거했습니다.
# print(f"⏳ '{LORA_ADAPTER_PATH}'에서 LoRA 어댑터를 로딩하여 모델에 적용합니다...")
# lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
# print("⏳ LoRA 가중치를 기본 모델에 병합합니다...")
# model = lora_model.merge_and_unload()

# ⭐️ 'model' 변수에 저장된 기본 모델을 파이프라인에 바로 사용합니다.
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
print("✅ 모델 로딩 및 설정이 완료되었습니다. (사전 학습된 기본 가중치 사용)")


# --- 6. 답변 후처리 및 유틸리티 (기존 코드와 동일) ---
def post_process_answer(generated_text: str, original_question: str) -> str:
    answer = generated_text.strip()
    if not answer: return "1"
    if "###" in answer: answer = answer.split("###")[0].strip()
        
    if is_multiple_choice(original_question):
        match = re.search(r'(?:정답은|답은|선택은)\s*\D*(\d+)', answer); 
        if match: return match.group(1)
        match = re.search(r'\b(\d+)\s*(?:번|번입니다|\.)', answer); 
        if match: return match.group(1)
        match = re.search(r"^\s*(\d+)", answer); 
        if match: return match.group(1)
        match = re.search(r'(\d+)', answer); 
        if match: return match.group(1)
        return "1"
    return answer if answer else "답변을 생성하지 못했습니다."

def is_code_detected(text: str) -> bool:
    code_keywords = ['def ', 'import ', 'class ', 'r\'', 'sys.stdout', 'ans_qna']
    return any(keyword in text.lower() for keyword in code_keywords)


# --- 7. 메인 실행 (RAG 적용) ---
if __name__ == "__main__":
    retriever = build_or_load_rag_backend_langchain()
    
    if not retriever:
        print("❌ RAG 백엔드 생성에 실패하여 추론을 중단합니다.")
        exit()

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    preds = []
    MAX_RETRIES = 3 

    for index, q in tqdm(enumerate(test_df['Question']), total=len(test_df), desc="🚀 추론 진행 중"):
        
        # LangChain Retriever를 사용하여 관련 문서 검색
        retrieved_docs = retriever.invoke(q)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
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

    print("\n 추론이 완료되었습니다. 제출 파일을 생성합니다...")
    try:
        sample_submission = pd.read_csv('/workspace/open/sample_submission.csv')
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 제출 파일 생성이 완료되었습니다: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print(f"❌ 오류: '/workspace/open/sample_submission.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
