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
LORA_ADAPTER_PATH = "./midm-lora-adapter-unified-trainer/checkpoint-201" 

# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_fewshot_v2.csv' # 저장될 파일 이름 변경

# --- 2. Few-shot 예시 및 프롬프트 템플릿 (★★★★★ 답변 형식 제어를 위해 지시문 강화) ---

# 모델에게 문제 유형별 모범 답안 형식을 명확히 알려주어 답변의 정확도와 일관성을 높입니다.
# 지시문을 "번호만 출력하시오", "답변 내용만 서술하시오" 와 같이 더 강력하게 수정했습니다.
FEW_SHOT_EXAMPLES = """
### 지시:
다음 질문에 대한 올바른 선택지의 번호만 출력하시오.

### 질문:
개인정보 보호법상, 정보주체의 동의 없이 개인정보를 제3자에게 제공할 수 있는 경우가 아닌 것은?
1. 정보주체로부터 별도의 동의를 받은 경우
2. 법률에 특별한 규정이 있거나 법령상 의무를 준수하기 위하여 불가피한 경우
3. 정보주체 또는 그 법정대리인이 의사표시를 할 수 없는 상태에 있거나 주소불명 등으로 사전 동의를 받을 수 없는 경우로서 명백히 정보주체 또는 제3자의 급박한 생명, 신체, 재산의 이익을 위하여 필요하다고 인정되는 경우
4. 통계작성 및 학술연구 등의 목적을 위하여 필요한 경우로서 특정 개인을 알아볼 수 없는 형태로 개인정보를 제공하는 경우

### 답변:
1

---

### 지시:
다음 질문에 대한 답변 내용만 서술하시오.

### 질문:
개인정보 보호법상 '가명처리'란 무엇인지 설명하시오.

### 답변:
개인정보의 일부를 삭제하거나 일부 또는 전부를 대체하는 등의 방법으로 추가 정보 없이는 특정 개인을 알아볼 수 없도록 처리하는 것.
"""

# --- 3. 유틸리티 함수 (변경 없음) ---

def is_multiple_choice(question_text: str) -> bool:
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> tuple[str, list[str]]:
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

# --- 4. 프롬프트 생성기 (지시문 강화 적용) ---

def make_prompt(text: str) -> str:
    """
    질문 유형에 따라 Few-shot 예시가 포함된 프롬프트를 생성합니다.
    강화된 지시문을 사용하여 모델이 답변 형식을 명확히 인지하도록 돕습니다.
    """
    # 1. 먼저 Few-shot 예시들을 프롬프트 앞부분에 추가합니다.
    base_prompt = FEW_SHOT_EXAMPLES.strip() + "\n\n---\n\n"
    
    # 2. 실제 질문에 대해 강화된 지시문을 적용합니다.
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        # 객관식: 정답 '번호만' 출력하도록 명시
        final_prompt = f"""### 지시:
다음 질문에 대한 올바른 선택지의 번호만 출력하시오.

### 질문:
{question}

### 선택지:
{chr(10).join(options)}

### 답변:"""
    else:
        # 주관식: '답변 내용만' 서술하도록 명시
        final_prompt = f"""### 지시:
다음 질문에 대한 답변 내용만 서술하시오.

### 질문:
{text}

### 답변:"""

    return base_prompt + final_prompt

# --- 5. 모델 및 토크나이저 로드 ---

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

# --- 6. 추론 및 후처리 (변경 없음) ---

def post_process_answer(generated_text: str, original_question: str) -> str:
    """생성된 텍스트에서 답변 부분만 추출하고 정리합니다."""
    answer = generated_text.strip()

    if not answer:
        return "미응답"

    if is_multiple_choice(original_question):
        match = re.match(r"\D*([1-9][0-9]?)", answer)
        return match.group(1) if match else "0"
    else:
        # 만약을 대비해 답변에 프롬프트 키워드가 포함된 경우 제거
        if "###" in answer:
            answer = answer.split("###")[0].strip()
        return answer

# --- 7. 메인 실행 ---
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
            max_new_tokens=256, # 주관식 답변을 위해 조금 더 여유있게 설정
            temperature=0.01,   # 일관된 답변 생성을 위해 매우 낮게 설정
            top_p=0.9,
            do_sample=True,
            return_full_text=False
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
