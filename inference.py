import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os

# --- 1. 설정 (Configuration) ---

# 기본 모델 ID (학습에 사용한 모델)
BASE_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"

# ⭐️ 사용자 설정: 학습된 LoRA 어댑터가 저장된 경로를 지정해주세요.
LORA_ADAPTER_PATH = "/workspace/2025-AI-Challeng-finance/midm-lora-adapter-unified-trainer/checkpoint-134" 

# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_no_fewshot.csv' # 저장될 파일 이름 변경

# --- 2. 유틸리티 함수 (변경 없음) ---

def is_multiple_choice(question_text: str) -> bool:
    """2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주합니다."""
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> tuple[str, list[str]]:
    """전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리합니다."""
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options

# --- 3. 프롬프트 생성기 (★★★★★ Few-shot 예시 제거 ★★★★★) ---

def make_prompt(text: str) -> str:
    """
    질문 유형에 따라 LoRA 학습에 사용된 기본 프롬프트를 생성합니다.
    """
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        # 객관식 프롬프트
        prompt = f"""### 지시:
다음 질문에 대한 올바른 답변을 선택지에서 고르시오.

### 질문:
{question}

### 선택지:
{chr(10).join(options)}

### 답변:"""
    else:
        # 주관식 프롬프트
        prompt = f"""### 지시:
다음 질문에 대해 핵심 키워드를 중심으로 서술하시오.

### 질문:
{text}

### 답변:"""

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

# --- 5. 추론 및 후처리 ---

def post_process_answer(generated_text: str, original_question: str) -> str:
    """생성된 텍스트에서 최종 답변을 추출하고 정리합니다."""
    answer = generated_text.strip()
    
    if not answer:
        return "미응답"

    # 만약을 대비해 답변에 프롬프트 키워드가 포함된 경우 제거
    if "###" in answer:
        answer = answer.split("###")[0].strip()
        
    if is_multiple_choice(original_question):
        match = re.match(r"\D*([1-9][0-9]?)", answer)
        return match.group(1) if match else "1" # 숫자 추출 실패 시 1번으로 추측
    
    return answer if answer else "답변을 생성하지 못했습니다."

# --- 6. 메인 실행 ---
if __name__ == "__main__":
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    preds = []
    for q in tqdm(test_df['Question'], desc="🚀 추론 진행 중"):
        prompt = make_prompt(q)
        
        output = pipe(
            prompt, 
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )
        
        generated_text = output[0]['generated_text']
        pred_answer = post_process_answer(generated_text, original_question=q)
        preds.append(pred_answer)

    print("📄 추론이 완료되었습니다. 제출 파일을 생성합니다...")
    try:
        sample_submission = pd.read_csv('/workspace/open/sample_submission.csv')
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 제출 파일 생성이 완료되었습니다: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print(f"❌ 오류: '/workspace/open/sample_submission.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
