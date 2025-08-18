import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os

# --- 1. 설정 (Configuration) ---

# ⭐️ 수정 1: 기본 모델 ID를 SOLAR로 변경
BASE_MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"

# ⭐️ 사용자 설정: 학습된 LoRA 어댑터가 저장된 경로를 지정해주세요.
# SOLAR 모델로 새로 학습한 LoRA 어댑터 경로를 사용해야 합니다.
LORA_ADAPTER_PATH = "/workspace/2025-AI-Challeng-finance/midm-lora-adapter-combined-laws/checkpoint-22" 

# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_solar_inference.csv' 

# --- 2. 유틸리티 함수 ---

def is_multiple_choice(question_text: str) -> bool:
    """질문이 객관식인지 판별합니다."""
    lines = question_text.strip().split("\n")
    # 숫자로 시작하고 공백이나 점으로 구분되는 선택지가 2개 이상일 경우 객관식으로 판단
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

# ⭐️ 수정 2: SOLAR 모델에 최적화된 프롬프트 템플릿으로 변경
def make_prompt(text: str) -> str:
    """
    질문 유형에 따라 SOLAR 모델에 최적화된 프롬프트를 생성합니다.
    """
    # SOLAR의 공식 프롬프트 템플릿: ### User:\n{instruction}\n\n### Assistant:\n
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        # 객관식 질문을 instruction으로 구성
        instruction = f"다음 질문에 대한 올바른 답변의 '번호'만 출력하세요.\n\n### 질문:\n{question}\n\n### 선택지:\n{chr(10).join(options)}"
    else:
        # 주관식 질문을 instruction으로 구성
        instruction = f"다음 질문에 대해 핵심 내용을 담아 완벽한 한국어 문장으로 서술하세요.\n\n### 질문:\n{text}"
    
    # 최종 프롬프트 반환
    return f"### User:\n{instruction}\n\n### Assistant:\n"

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
    print(f"상세 오류: {e}")
    # SOLAR 모델은 LoRA 없이도 성능이 좋으므로, 어댑터 로딩 실패 시 기본 모델로 계속 진행하도록 설정
    print("⚠️ 경고: LoRA 어댑터 로딩에 실패하여 기본 모델로 추론을 진행합니다.")
    model = base_model 

if isinstance(model, PeftModel):
    print("⏳ LoRA 가중치를 기본 모델에 병합합니다...")
    model = model.merge_and_unload()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("✅ 모델 로딩 및 설정이 완료되었습니다.")

# --- 4. 추론 및 후처리 ---

# ⭐️ 수정 3: 후처리 로직 강화
def post_process_answer(generated_text: str, original_question: str) -> str:
    """생성된 텍스트에서 최종 답변을 추출하고 정리하는 강화된 함수입니다."""
    answer = generated_text.strip()
    
    if not answer:
        return "1" if is_multiple_choice(original_question) else "답변을 생성하지 못했습니다."

    # 답변에 프롬프트 키워드가 포함된 경우 제거 (예: ### Assistant:)
    if "###" in answer:
        answer = answer.split("###")[0].strip()
        
    if is_multiple_choice(original_question):
        # 1단계: "정답은 5", "답은 5번" 등 명확한 패턴에서 숫자 추출
        match = re.search(r'(?:정답은|답은|선택은|답변은)\s*\D*(\d+)', answer)
        if match:
            return match.group(1)

        # 2단계: "5번", "5." 와 같은 패턴에서 숫자 추출
        match = re.search(r'\b(\d+)\s*(?:번|번입니다|\.)', answer)
        if match:
            return match.group(1)

        # 3단계: 문장 맨 앞에 있는 숫자 추출
        match = re.search(r"^\s*(\d+)", answer)
        if match:
            return match.group(1)

        # 4단계: 위 모든 조건에 해당하지 않을 경우, 텍스트 전체에서 처음 발견되는 숫자 추출
        match = re.search(r'(\d+)', answer)
        if match:
            return match.group(1)
            
        # 5단계: 그래도 숫자를 찾지 못하면 기본값 '1' 반환
        return "1"
    
    return answer if answer else "답변을 생성하지 못했습니다."


def is_code_detected(text: str) -> bool:
    """간단한 키워드 기반으로 생성된 텍스트에 코드가 포함되었는지 확인합니다."""
    code_keywords = ['def ', 'import ', 'class ', 'r\'', 'sys.stdout', 'ans_qna']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in code_keywords)


# --- 5. 메인 실행 ---
if __name__ == "__main__":
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    preds = []
    MAX_RETRIES = 3 

    for index, q in tqdm(enumerate(test_df['Question']), total=len(test_df), desc="🚀 추론 진행 중"):
        prompt = make_prompt(q)
        
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

