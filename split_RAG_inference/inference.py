import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. 설정 (Configuration) ---
# ⭐️ 사용자 설정: 경로들을 본인 환경에 맞게 확인해주세요.
BASE_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
LORA_ADAPTER_PATH = "/workspace/checkpoint-22" 
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_optimized.csv' 

# ⭐️ RAG DB 설정
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
FAISS_DB_PATH = "/workspace/2025-AI-Challeng-finance/split_RAG_inference/faiss_index_laws"


# --- 2. RAG DB 로드 전용 함수 ---
def load_rag_retriever():
    """
    미리 생성된 FAISS 벡터 DB를 로드하여 Retriever를 반환합니다.
    """
    if not os.path.exists(FAISS_DB_PATH):
        print(f"❌ 오류: 벡터 DB 경로 '{FAISS_DB_PATH}'를 찾을 수 없습니다.")
        exit()

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"⏳ 기존 벡터 DB를 '{FAISS_DB_PATH}'에서 로드합니다...")
    vector_db = FAISS.load_local(
        FAISS_DB_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print("✅ 벡터 DB 로드 완료.")
    
    return vector_db.as_retriever(search_kwargs={'k': 3})


# --- 3. 프롬프트 및 유틸리티 함수 (변경 없음) ---
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
    return " ".join(q_lines), options

def make_prompt(text: str) -> str:
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = f"### 지시:\n다음 질문에 대한 올바른 답변의 '번호'만 출력하세요.\n\n### 질문:\n{question}\n\n### 선택지:\n{chr(10).join(options)}\n\n### 답변:\n"
    else:
        prompt = f"### 지시:\n다음 질문에 대해 핵심 키워드를 중심으로 완벽한 한국어 문장으로 서술하세요.\n\n### 질문:\n{text}\n\n### 답변:\n"
    return prompt

def make_rag_prompt(text: str, context: str) -> str:
    if not context.strip():
        return make_prompt(text)
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = f"### 지시:\n주어진 '참고 문서'를 바탕으로 다음 질문에 대한 올바른 답변의 '번호'만 출력하세요.\n\n### 참고 문서:\n{context}\n\n### 질문:\n{question}\n\n### 선택지:\n{chr(10).join(options)}\n\n### 답변:\n"
    else:
        prompt = f"### 지시:\n주어진 '참고 문서'의 내용을 바탕으로 다음 질문에 대해 완벽한 한국어 문장으로 서술하세요.\n\n### 참고 문서:\n{context}\n\n### 질문:\n{text}\n\n### 답변:\n"
    return prompt

def post_process_answer(generated_text: str, original_question: str) -> str:
    answer = generated_text.strip().split("###")[0].strip()
    if is_multiple_choice(original_question):
        match = re.search(r"^\s*(\d+)", answer)
        return match.group(1) if match else "1"
    return answer if answer else "답변을 생성하지 못했습니다."


# --- 4. LLM 로드 (Flash Attention 2 적용) ---
print("⏳ 추론용 LLM을 로딩합니다...")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

# ⭐️ Flash Attention 2 적용
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    quantization_config=quantization_config, 
    device_map="auto",
    attn_implementation="flash_attention_2"  # 이 부분 추가
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"⏳ '{LORA_ADAPTER_PATH}'에서 LoRA 어댑터를 적용합니다...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model = model.merge_and_unload()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
print("✅ 추론용 LLM 로딩 완료.")


# --- 5. 메인 실행 (배치 처리 적용) ---
if __name__ == "__main__":
    retriever = load_rag_retriever()

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다.")
        exit()

    # ⭐️ 배치 추론 실행
    preds = []
    batch_size = 8  # 한 번에 처리할 질문 수 (GPU 메모리에 따라 4, 8, 16 등으로 조절)
    all_questions = test_df['Question'].tolist()

    for i in tqdm(range(0, len(all_questions), batch_size), desc="🚀 배치 추론 진행 중"):
        batch_questions = all_questions[i:i + batch_size]
        
        # 1. RAG 검색 (배치)
        batch_contexts = []
        for q in batch_questions:
            try:
                retrieved_docs = retriever.get_relevant_documents(q)
                context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                batch_contexts.append(context_text)
            except Exception as e:
                print(f"⚠️ RAG 검색 중 오류 발생: {e}")
                batch_contexts.append("")

        # 2. 프롬프트 생성 (배치)
        batch_prompts = [make_rag_prompt(q, ctx) for q, ctx in zip(batch_questions, batch_contexts)]

        # 3. LLM 추론 (배치)
        outputs = pipe(
            batch_prompts, 
            max_new_tokens=512, 
            do_sample=True, 
            temperature=0.1, 
            return_full_text=False, 
            eos_token_id=tokenizer.eos_token_id,
            batch_size=batch_size # 파이프라인에 배치 크기 명시
        )
        
        # 4. 결과 후처리
        for idx, output in enumerate(outputs):
            # 'outputs'는 리스트의 리스트 형태일 수 있으므로 확인 후 처리
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0]['generated_text']
            elif isinstance(output, dict):
                generated_text = output['generated_text']
            else:
                generated_text = "" # 예외 처리

            pred_answer = post_process_answer(generated_text, original_question=batch_questions[idx])
            preds.append(pred_answer)

    # 제출 파일 생성
    try:
        sample_submission = pd.read_csv('/workspace/open/sample_submission.csv')
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\n✅ 제출 파일 생성 완료: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print("❌ 오류: 'sample_submission.csv' 파일을 찾을 수 없습니다.")
