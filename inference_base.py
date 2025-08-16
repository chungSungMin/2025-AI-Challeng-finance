import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import os

# --- 1. 설정 (Configuration) ---

# 기본 모델 ID (사용할 모델)
BASE_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"

# 테스트 데이터 및 제출 파일 경로
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_cot_model.csv' 

# --- 2. 유틸리티 함수 ---

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

# (핵심 변경) 1. make_prompt 함수를 Chain-of-Thought 방식으로 수정
def make_prompt(text: str) -> str:
    """
    질문 유형에 따라 Chain-of-Thought 프롬프트를 생성합니다.
    """
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        # 객관식 CoT 프롬프트
        prompt = f"""### 지시:
                    다음 질문에 대해 각 선택지를 분석하고, 단계별 추론 과정을 서술한 뒤, 최종적으로 반드시 가장 올바른 답변의 번호를 '최종 답변: [번호]' 형식으로 출력하세요.

                    ### 예시:
                    질문: 금융기관의 개인정보 보호 의무를 가장 잘 설명하는 것은 무엇인가?
                    선택지:
                    1. 고객의 동의 없이 개인정보를 수집할 수 없다.
                    2. 고객의 동의 없이 개인정보를 처리할 수 없다.
                    3. 고객의 동의 없이 개인정보를 공개할 수 없다.
                    4. 고객의 동의 없이 개인정보를 삭제할 수 없다.
                    답변:
                    * 1번 고객의 동의 없이 개인정보를 수집하는 것은 금융기관의 기본적인 의무이며, 이는 개인정보 보호법에 명시되어 있다.
                    * 2번 고객의 동의 없이 개인정보를 처리하는 것은 가능하나, 해당 처리의 범위와 목적이 명확해야 한다.
                    * 3번 고객의 동의 없이 개인정보를 공개하는 것은 불가능하다
                    * 4번 "고객의 동의 없이 개인정보를 삭제하는 것은 불가능하다.
                    따라서 가장 올바른 설명은 1번이다.
                    최종 답변: 1



                    ### 예시:
                    질문: 정보통신망 이용자로부터 개인정보를 수집할 때, 수집 목적을 명확히 해야 하는 이유는 무엇인가?
                    선택지:
                    1. 정보통신망 이용자의 동의를 얻기 위한 절차
                    2. 법적 요구 사항을 충족하기 위해 필요한 절차 
                    3. 정보통신망 서비스 제공자가 개인정보를 안전하게 관리하기 위한 방법
                    4. 정보통신망 이용자에게 개인정보 처리 방침을 알리는 의무
                    답변:
                    * 1번 정보통신망 이용자로부터 개인정보를 수집할 때, 수집 목적을 명확히 해야 하므로 이용자는 자신의 정보가 어떻게 사용되는지 이해할 수 있습니다.
                    * 2번 법적 요구 사항을 충족하기 위해서는 수집 목적을 명확히 해야 하지만, 이는 개인정보 수집의 기본 원칙이지, 수집 목적을 명확히 하지 않아도 되는 이유가 아닙니다.
                    * 3번 정보통신망 서비스 제공자가 개인정보를 안전하게 관리하기 위한 방법은 중요하지만, 수집 목적을 명확히 하는 것은 별개의 문제입니다.
                    * 4번 정보통신망 이용자에게 개인정보 처리 방침을 알리는 의무는 있지만, 이는 수집 목적을 명확히 하는 것과는 다릅니다.
                    따라서 가장 올바른 설명은 1번이다.
                    최종 답변: 1


                    ### 실제 입력
                    질문:
                    {question}

                    선택지:
                    {chr(10).join(options)}

                    답변:
                """
    else:
        # 주관식 프롬프트 (기존 방식 유지하되 지시문 통일)
        prompt = f"""### 지시:
                    다음 질문에 대해 핵심 키워드를 중심으로 완벽한 문장으로 서술하세요.

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

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("✅ 모델 로딩 및 설정이 완료되었습니다.")

# (핵심 변경) 2. post_process_answer 함수를 CoT 결과에 맞게 수정
def post_process_answer(generated_text: str, original_question: str) -> str:
    """생성된 텍스트에서 최종 답변을 추출하고 정리합니다."""
    answer = generated_text.strip()
    
    if not answer:
        return "미응답"

    if is_multiple_choice(original_question):
        # CoT 결과에서 '최종 답변: [숫자]' 패턴을 찾아 숫자를 추출합니다.
        match = re.search(r"최종 답변:\s*([1-9][0-9]?)", answer)
        if match:
            return match.group(1)
        else:
            # 만약 '최종 답변' 패턴을 찾지 못하면, 원래 방식대로 답변에서 숫자라도 찾아봅니다.
            fallback_match = re.search(r"\D*([1-9][0-9]?)", answer)
            return fallback_match.group(1) if fallback_match else "1" # 최후의 수단으로 1번 추측
    
    # 주관식 답변은 기존 방식대로 간단히 정리
    if "###" in answer:
        answer = answer.split("###")[0].strip()
        
    return answer if answer else "답변을 생성하지 못했습니다."

# --- 6. 메인 실행 ---
if __name__ == "__main__":
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ '{TEST_CSV_PATH}'에서 테스트 데이터를 성공적으로 로드했습니다.")
    except FileNotFoundError:
        # 올바른 변수명으로 수정
        print(f"❌ 오류: '{TEST_CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()


    prompts = [make_prompt(q) for q in tqdm(test_df['Question'], desc="프롬프트 생성 중")]
    
    # 배치 처리를 위한 파이프라인 호출
    outputs = pipe(
        prompts, 
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
        batch_size=8 # GPU 메모리에 맞춰 배치 사이즈 조절
    )

    preds = []
    for i, output in enumerate(tqdm(outputs, desc="📄 결과 후처리 중")):
        generated_text = output[0]['generated_text']
        original_question = test_df['Question'].iloc[i]
        pred_answer = post_process_answer(generated_text, original_question)
        preds.append(pred_answer)

    print("📄 추론이 완료되었습니다. 제출 파일을 생성합니다...")
    try:
        sample_submission = pd.read_csv('/workspace/open/sample_submission.csv')
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ 제출 파일 생성이 완료되었습니다: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print(f"❌ 오류: '/workspace/open/sample_submission.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")