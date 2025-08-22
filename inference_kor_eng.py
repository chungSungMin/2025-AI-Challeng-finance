# import os
# import torch
# import pandas as pd
# import re
# from tqdm import tqdm
# import json

# # --- 1. 라이브러리 Import ---
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# from peft import PeftModel

# # --- 2. 경로 및 모델 설정 ---
# # 기본 모델 ID
# BASE_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
# # 학습된 LoRA 어댑터 경로
# LORA_ADAPTER_PATH = "/workspace/checkpoint-708" 
# # 테스트 데이터 및 제출 파일 경로
# TEST_CSV_PATH = '/workspace/open/test.csv'
# SAMPLE_SUBMISSION_PATH = '/workspace/open/sample_submission.csv'
# SUBMISSION_CSV_PATH = './submission_with_loaded_rag.csv' 

# # ★★★ 수정된 부분 1: 불러올 DB 경로와 임베딩 모델 지정 ★★★
# # 미리 구축한 FAISS DB가 저장된 폴더 경로
# FAISS_DB_PATH = "/workspace/2025-AI-Challeng-finance/faiss_db_kor_eng" 
# # DB 구축 시 사용했던 임베딩 모델 (DB와 반드시 일치해야 함)
# EMBEDDING_MODEL_NAME = "BAAI/bge-m3" 


# # --- 3. RAG 백엔드 로드 함수 (수정된 부분) ---
# def load_rag_retriever(db_path: str, embedding_model_name: str):
#     """지정된 경로에서 미리 구축된 FAISS 벡터 DB를 로드하여 Retriever를 반환합니다."""
    
#     if not os.path.exists(db_path):
#         print(f"❌ 치명적 오류: RAG 데이터베이스를 '{db_path}'에서 찾을 수 없습니다.")
#         print("DB 경로를 확인하거나, DB를 먼저 구축해주세요.")
#         return None

#     print(f"⏳ 기존 벡터 DB를 '{db_path}'에서 로드합니다...")
#     try:
#         embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
#         vector_db = FAISS.load_local(
#             db_path, 
#             embedding_model, 
#             allow_dangerous_deserialization=True
#         )
#         print("✅ 벡터 DB 로드 완료.")
#         # LangChain의 Retriever 객체 반환
#         return vector_db.as_retriever(search_kwargs={'k': 3})
#     except Exception as e:
#         print(f"❌ 오류: 벡터 DB 로딩 중 오류 발생: {e}")
#         return None

# # --- 4. 유틸리티 및 프롬프트 함수 (기존과 동일) ---
# def is_multiple_choice(question_text: str) -> bool:
#     lines = question_text.strip().split("\n")
#     option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?[\.\s]", line)) for line in lines)
#     return option_count >= 2

# def extract_question_and_choices(full_text: str) -> tuple[str, list[str]]:
#     lines = full_text.strip().split("\n")
#     q_lines, options = [], []
#     for line in lines:
#         if re.match(r"^\s*[1-9][0-9]?[\.\s]", line):
#             options.append(line.strip())
#         else:
#             q_lines.append(line.strip())
#     question = " ".join(q_lines)
#     return question, options

# def make_rag_prompt(text: str, context: str) -> str:
#     """RAG 검색 결과를 우선적으로 사용하여 프롬프트를 생성합니다."""
#     if is_multiple_choice(text):
#         question, options = extract_question_and_choices(text)
#         prompt = f"""### 지시:
# 주어진 **'참고 문서'의 내용만을 근거**로 하여 다음 질문에 대한 올바른 답변의 '번호'만 출력하세요. 다른 설명은 절대 추가하지 마세요.

# ### 참고 문서:
# {context}

# ### 질문:
# {question}

# ### 선택지:
# {chr(10).join(options)}

# ### 답변:
# """
#     else:
#         prompt = f"""### 지시:
# 주어진 **'참고 문서'의 내용만을 근거**로 하여 '질문'에 답변하세요.
# 문서에서 질문과 관련된 핵심 내용을 종합하여, 전문 용어를 사용해 2~3개의 완벽한 한국어 문장으로 설명해야 합니다.
# **'참고 문서에 따르면'과 같은 표현은 절대 사용하지 마세요.** 당신의 배경 지식이나 외부 정보는 사용하지 마세요.

# ### 참고 문서:
# {context}

# ### 질문:
# {text}

# ### 답변:
# """
#     return prompt

# # --- 5. 모델 및 토크나이저 로드 (기존과 동일) ---
# print("⏳ 모델과 토크나이저를 로딩합니다...")
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )
# base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=quantization_config, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# print(f"⏳ '{LORA_ADAPTER_PATH}'에서 LoRA 어댑터를 로딩하여 모델에 적용합니다...")
# model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
# print("⏳ LoRA 가중치를 기본 모델에 병합합니다...")
# model = model.merge_and_unload()
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# print("✅ 모델 로딩 및 설정이 완료되었습니다.")

# # --- 6. 답변 후처리 (기존과 동일) ---
# def post_process_answer(generated_text: str, original_question: str) -> str:
#     answer = generated_text.strip()
#     if not answer: return "1"
#     if "###" in answer: answer = answer.split("###")[-1].strip()
#     if is_multiple_choice(original_question):
#         match = re.search(r'(\d+)', answer)
#         return match.group(1) if match else "1"
#     return answer if answer else "답변을 생성하지 못했습니다."

# # --- 7. 메인 실행 (수정된 부분) ---
# if __name__ == "__main__":
#     # ★★★ 수정된 부분 2: DB 로드 함수 호출 ★★★
#     retriever = load_rag_retriever(FAISS_DB_PATH, EMBEDDING_MODEL_NAME)
    
#     if not retriever:
#         print("❌ RAG 백엔드 로딩에 실패하여 추론을 중단합니다.")
#         exit()

#     try:
#         test_df = pd.read_csv(TEST_CSV_PATH)
#         print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
#     except FileNotFoundError:
#         print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
#         exit()

#     preds = []
#     for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="🚀 추론 진행 중"):
#         question = row['Question']
        
#         # LangChain Retriever를 사용하여 관련 문서 검색
#         retrieved_docs = retriever.invoke(question)
#         context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
#         prompt = make_rag_prompt(question, context_text)
        
#         output = pipe(
#             prompt, 
#             max_new_tokens=512,
#             temperature=0.1,
#             top_p=0.9,
#             do_sample=True,
#             return_full_text=False,
#             eos_token_id=tokenizer.eos_token_id
#         )
#         generated_text = output[0]['generated_text']

#         pred_answer = post_process_answer(generated_text, original_question=question)
#         preds.append(pred_answer)

#     print("\n📄 추론이 완료되었습니다. 제출 파일을 생성합니다...")
#     try:
#         sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
#         sample_submission['Answer'] = preds
#         sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
#         print(f"✅ 제출 파일 생성이 완료되었습니다: '{SUBMISSION_CSV_PATH}'")
#     except FileNotFoundError:
#         print(f"❌ 오류: '{SAMPLE_SUBMISSION_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")





import os
import torch
import pandas as pd
import re
from tqdm import tqdm
import json

# --- 1. 라이브러리 Import ---
from langchain.retrievers import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

# --- 2. 경로 및 모델 설정 ---
# 기본 모델 ID
BASE_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
# 학습된 LoRA 어댑터 경로
LORA_ADAPTER_PATH = "/workspace/checkpoint-708" 
# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SAMPLE_SUBMISSION_PATH = '/workspace/open/sample_submission.csv'
SUBMISSION_CSV_PATH = './submission_multiquery_rag.csv' 

# 미리 구축한 FAISS DB가 저장된 폴더 경로
FAISS_DB_PATH = "/workspace/2025-AI-Challeng-finance/faiss_db_kor_eng" 
# DB 구축 시 사용했던 임베딩 모델 (DB와 반드시 일치해야 함)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3" 


# --- 3. MultiQueryRetriever 로드 함수 ---
def load_multi_query_retriever(db_path: str, embedding_model_name: str, llm_pipeline):
    """
    미리 구축된 FAISS DB와 LLM을 사용하여 MultiQueryRetriever를 생성합니다.
    """
    if not os.path.exists(db_path):
        print(f"❌ 치명적 오류: RAG 데이터베이스를 '{db_path}'에서 찾을 수 없습니다.")
        return None

    print(f"⏳ 기존 벡터 DB를 '{db_path}'에서 로드합니다...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_db = FAISS.load_local(
            db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ 벡터 DB 로드 완료.")

        # LangChain에서 사용할 수 있도록 LLM 파이프라인 객체화
        llm_for_retriever = HuggingFacePipeline(pipeline=llm_pipeline)
        
        # MultiQueryRetriever 생성
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(search_kwargs={'k': 5}), # 후보군을 늘리기 위해 k값 상향
            llm=llm_for_retriever
        )
        print("✅ MultiQueryRetriever 생성 완료.")
        return multi_query_retriever
    except Exception as e:
        print(f"❌ 오류: 벡터 DB 로딩 또는 Retriever 생성 중 오류 발생: {e}")
        return None

# --- 4. 유틸리티 및 프롬프트 함수 ---
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
    """RAG 검색 결과를 사용하여 프롬프트를 생성합니다."""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = f"""### 지시:
주어진 **'참고 문서'의 내용만을 근거**로 하여 다음 질문에 대한 올바른 답변의 '번호'만 출력하세요. 다른 설명은 절대 추가하지 마세요.

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
**'참고 문서에 따르면'과 같은 표현은 절대 사용하지 마세요.**

### 참고 문서:
{context}

### 질문:
{text}

### 답변:
"""
    return prompt

# --- 5. 모델 및 토크나이저 로드 ---
print("⏳ 모델과 토크나이저를 로딩합니다...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=quantization_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"⏳ '{LORA_ADAPTER_PATH}'에서 LoRA 어댑터를 로딩하여 모델에 적용합니다...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
print("⏳ LoRA 가중치를 기본 모델에 병합합니다...")
model = model.merge_and_unload()
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("✅ 모델 로딩 및 설정이 완료되었습니다.")

# --- 6. 답변 후처리 ---
def post_process_answer(generated_text: str, original_question: str) -> str:
    answer = generated_text.strip()
    if not answer: return "1"
    if "###" in answer: answer = answer.split("###")[-1].strip()
    if is_multiple_choice(original_question):
        match = re.search(r'(\d+)', answer)
        return match.group(1) if match else "1"
    return answer if answer else "답변을 생성하지 못했습니다."

# --- 7. 메인 실행 ---
if __name__ == "__main__":
    # 모델과 파이프라인이 먼저 로드된 후 Retriever를 생성합니다.
    retriever = load_multi_query_retriever(FAISS_DB_PATH, EMBEDDING_MODEL_NAME, pipe)
    
    if not retriever:
        print("❌ RAG 백엔드 로딩에 실패하여 추론을 중단합니다.")
        exit()

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="� 추론 진행 중"):
        question = row['Question']
        
        # MultiQueryRetriever를 사용하여 관련 문서 검색
        retrieved_docs = retriever.invoke(question)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt = make_rag_prompt(question, context_text)
        
        output = pipe(
            prompt, 
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = output[0]['generated_text']

        pred_answer = post_process_answer(generated_text, original_question=question)
        preds.append(pred_answer)

    print("\n📄 추론이 완료되었습니다. 제출 파일을 생성합니다...")
    try:
        sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 제출 파일 생성이 완료되었습니다: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print(f"❌ 오류: '{SAMPLE_SUBMISSION_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
