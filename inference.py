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
LORA_ADAPTER_PATH = "/workspace/2025-AI-Challeng-finance/midm-lora-adapter-combined-laws/checkpoint-22" 

# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_new_traindataset.csv' 

# --- 2. 유틸리티 함수 (변경 없음) ---

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


def make_prompt(text: str) -> str:
    """
    질문 유형에 따라 LoRA 학습에 사용된 기본 프롬프트를 생성합니다.
    """
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        # 객관식 프롬프트
        prompt = f"""### 지시:
                    다음 질문에 대한 올바른 답변을 선택지에서 고르세요.

                    ### 예시1
                    질문 : 개인정보보호법에 따라 보호위원회가 과징금을 부과할 때 고려해야 하는 사항 중 하나는 무엇인가요?
                    선택지 : 
                    1. 위반행위의 내용 및 정도
                    2. 개인정보 처리자의 주소와 연락처
                    3. 개인정보의 수집 및 이용 목적
                    4. 개인정보 처리자의 직원 수와 연봉 수준
                    답 : 1

                    ### 예시2
                    질문 : 개인정보 보호 기본계획은 언제까지 수립해야 하는가?
                    선택지 : 
                    1. 개인정보 보호 기본계획은 매년 12월 31일까지 수립해야 한다
                    2. 그 3년이 시작되는 해의 전년도 6월 30일까지 수립해야한다.
                    3. 개인정보 보호 기본계획은 매년 9월 30일까지 수립해야 한다.
                    4. 개인정보 보호 기본계획은 그 3년이 시작되는 해의 전년도 12월 31일까지 수립해야 한다.
                    5. 개인정보 보호 기본계획은 매년 10월 30일까지 수립해야 한다.
                    답변 : 2

                    ### 실제 입력
                    질문:
                    {question}

                    선택지:
                    {chr(10).join(options)}

                    답변:
                """
    else:
        # 주관식 프롬프트
        prompt = f"""### 지시:
                    다음 질문에 대해 핵심 키워드를 중심으로 알고있는 대로 서술하세요.
                    단, 모든 문장은 한국어로 구성되어야 하고, 완벽한 문장으로 서술해야합니다. 
                    

                    ### 예시1 : 
                    질문 : 개인정보의 국외 이전이 중지될 수 있는 조건은 무엇인가?
                    답변 : 개인정보의 국외 이전이 중지될 수 있는 조건은 제28조의8제1항, 제4항 또는 제5항을 위반하거나 개인정보를 이전받는 자나 국가가 개인정보 보호 수준에 미치지 못하여 정보주체에게 피해가 발생할 우려가 있는 경우이다.


                    ### 예시2 : 
                    질문 : 분쟁조정위원회가 분쟁조정 신청을 받은 후 심사하여 조정안을 작성해야 하는 기간은 얼마인가?
                    답변 : 분쟁조정위원회는 분쟁조정 신청을 받은 날부터 60일 이내에 심사하여 조정안을 작성해야 한다.
                    

                    ### 실제 입력 : 
                    질문:
                    {text}

                    ### 답변:
                """

    return prompt

# --- 4. 모델 및 토크나이저 로드 ---

print("⏳ 모델과 토크나이저를 로딩합니다...")

#=============양자화=====================#

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

# trust_remote_code : hugging face에서 제공하는 추가적인 토크나이저 코드 실행 가능하도록 
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
    if "##" or   "###" or "---" in answer:
        answer = answer.split("###")[0].strip()
        
    if is_multiple_choice(original_question):
        match = re.match(r"\D*([1-9][0-9]?)", answer)
        return match.group(1) if match else "1" # 숫자 추출 실패 시 1번으로 추측
    
    return answer if answer else "답변을 생성하지 못했습니다."


def is_code_detected(text: str) -> bool:
    """간단한 키워드 기반으로 생성된 텍스트에 코드가 포함되었는지 확인합니다."""
    # 사용자가 제공한 특정 코드 패턴 및 일반적인 키워드 추가
    code_keywords = ['def ', 'import ', 'class ', 'r\'', 'sys.stdout', 'ans_qna']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in code_keywords)


# --- 6. 메인 실행 (재시도 로직 적용) ---
if __name__ == "__main__":
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    preds = []
    
    # ⭐️ 2. 최대 재시도 횟수 설정
    MAX_RETRIES = 3 

    for index, q in tqdm(enumerate(test_df['Question']), total=len(test_df), desc="🚀 추론 진행 중"):
        prompt = make_prompt(q)
        
        is_valid_answer = False
        retries = 0
        generated_text = ""

        # ⭐️ 3. 유효한 답변을 얻거나 최대 재시도 횟수에 도달할 때까지 반복
        while not is_valid_answer and retries < MAX_RETRIES:
            if retries > 0:
                print(f"\n🔄 TEST_{index} 질문에 대한 답변 재시도 중... ({retries}/{MAX_RETRIES})")

            output = pipe(
                prompt, 
                max_new_tokens=512,
                # ⭐️ 재시도할 때마다 temperature를 약간 높여 다른 답변을 유도
                temperature=0.1 + (retries * 0.15),
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = output[0]['generated_text']

            # 생성된 텍스트에 코드 패턴이 있는지 확인
            if is_code_detected(generated_text):
                retries += 1
                # 마지막 재시도에서도 실패하면, 경고 메시지 출력 후 루프 종료
                if retries == MAX_RETRIES:
                    print(f"❌ TEST_{index} 질문에 대해 최대 재시도 횟수 초과. 마지막으로 생성된 답변을 사용합니다.")
                    is_valid_answer = True # 루프를 빠져나가기 위해 True로 설정
            else:
                is_valid_answer = True # 코드 패턴이 없으므로 유효한 답변으로 간주하고 루프 종료

        # 최종적으로 얻은 답변을 후처리
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


