import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os

# --- 1. 설정 (Configuration) ---

# 기본 모델 ID
BASE_MODEL_ID = "dnotitia/DNA-2.0-14B"

# ⭐️ 사용자 설정: 'dnotitia/DNA-2.0-14B'로 학습된 LoRA 어댑터 경로
LORA_ADAPTER_PATH = "/workspace/2025-AI-Challeng-finance/dna-lora-adapter-trainer/checkpoint-500"

# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_batch_dna_inference.csv' # 파일 이름 변경


# --- 2. 유틸리티 및 프롬프트 함수 (RAG 관련 함수 제거) ---

def is_multiple_choice(question_text: str) -> bool:
    """질문이 객관식인지 판별합니다."""
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?[\.\s]", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> tuple[str, list[str]]:
    """객관식 질문에서 순수 질문과 선택지를 분리합니다."""
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

def make_prompt(text: str) -> str:
    """모델의 내부 지식만으로 답변을 생성하는 프롬프트입니다."""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = f"""### 지시:
다음 질문에 대한 올바른 답변의 '번호'만 출력하세요. 다른 설명은 절대 추가하지 마세요.

### 질문:
{question}

### 선택지:
{chr(10).join(options)}

### 답변:
"""
    else:
        prompt = f"""### 지시:
다음 질문에 대해 핵심 키워드를 중심으로 완벽한 한국어 문장으로 서술하세요. 배경 지식을 활용해서 답해주세요. "문서에 따르면~ " 이라는 내용을 쓰지 말아주세요.
최대한 **전문 용어**를 활용해서 서술해주세요. 그리고 마크다운을 사용하지말고, 2~3문장으로 핵심을 담아 서술하세요.

### 질문:
{text}

### 답변:
"""
    return prompt

# --- 3. 모델 및 토크나이저 로드 ---

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

tokenizer.padding_side = 'left'

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("✅ 모델 로딩 및 설정이 완료되었습니다.")


def post_process_answer(generated_text: str, original_question: str) -> str:
    """생성된 텍스트에서 최종 답변을 추출하고 정리하는 함수입니다."""
    answer = generated_text.strip()
    
    if not answer:
        return "1"

    if "###" in answer:
        answer = answer.split("###")[0].strip()
        
    if is_multiple_choice(original_question):
        match = re.search(r'(?:정답은|답은|선택은)\s*\D*(\d+)', answer)
        if match: return match.group(1)

        match = re.search(r'\b(\d+)\s*(?:번|번입니다|\.)', answer)
        if match: return match.group(1)

        match = re.search(r"^\s*(\d+)", answer)
        if match: return match.group(1)

        match = re.search(r'(\d+)', answer)
        if match: return match.group(1)
            
        return "1"
    
    return answer if answer else "답변을 생성하지 못했습니다."

# --- 4. 메인 실행 (배치 처리 적용) ---
# --- 4. 메인 실행 (수동 배치 처리 적용) ---
if __name__ == "__main__":
    print("[INFO] 모델의 내부 지식만으로 추론을 진행합니다.")

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    prompts = [make_prompt(q) for q in test_df['Question']]
    
    # ★★★ 수동 배치를 위한 설정 ★★★
    batch_size = 2
    all_outputs = [] # 모든 결과를 저장할 리스트

    print(f"🚀 배치 추론을 시작합니다 (배치 크기: {batch_size})...")
    
    # ★★★ tqdm으로 전체 루프를 감싸고, 수동으로 배치를 생성하여 파이프라인에 전달 ★★★
    for i in tqdm(range(0, len(prompts), batch_size), desc="🚀 추론 진행 중"):
        # 현재 처리할 배치 슬라이싱
        batch_prompts = prompts[i:i + batch_size]
        
        # 파이프라인은 현재 배치만 처리
        # batch_size 인자는 파이프라인 호출에서 제거 (수동으로 제어하므로)
        outputs_batch = pipe(
            batch_prompts,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )
        # 처리된 배치 결과를 전체 결과 리스트에 추가
        all_outputs.extend(outputs_batch)

    print("\n📄 추론 완료. 답변을 후처리합니다...")
    
    # 변수 이름을 outputs -> all_outputs 로 변경
    preds = []
    for i, output in enumerate(tqdm(all_outputs, desc="답변 후처리 중")):
        generated_text = output[0]['generated_text']
        original_question = test_df['Question'][i]
        pred_answer = post_process_answer(generated_text, original_question)
        preds.append(pred_answer)

    print("\n📄 제출 파일을 생성합니다...")
    try:
        sample_submission = pd.read_csv('/workspace/open/sample_submission.csv')
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 제출 파일 생성이 완료되었습니다: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print(f"❌ 오류: '/workspace/open/sample_submission.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")