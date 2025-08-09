import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def setup_model():
    """
    모델과 토크나이저를 설정하고 로드합니다.
    4비트 양자화를 사용하여 메모리 효율성을 높입니다.
    """
    print("모델 및 토크나이저 로드를 시작합니다...")
    
    # 4비트 양자화 설정: 메모리 사용량을 줄여 더 낮은 사양의 GPU에서도 실행 가능하게 함
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = "SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx"

    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        tokenizer.pad_token = tokenizer.eos_token

        # 모델 로드 (양자화 설정 적용)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # 사용 가능한 장치(GPU/CPU)에 모델을 자동으로 할당
            quantization_config=quantization_config,
        )
        print("✅ 모델 및 토크나이저 로드 완료.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 모델 로드 중 오류가 발생했습니다: {e}")
        print("인터넷 연결을 확인하거나 모델 ID가 올바른지 확인해주세요.")
        return None, None

def get_prompt_template():
    return """
        당신은 정보보안기사 국가공인시험 출제위원입니다. 수험생의 허를 찌르는 변별력 높은 문제를 만들어주세요.
        쉬운 문제부터 어려운 문제 까지 다양한 난이도의 문제를 생성해주세요. 

        [기본 규칙]
        1. 질문은 명확하고 오해의 소지가 없어야 합니다.
        2. 4개의 선지를 생성하며, 그중 정답은 반드시 하나여야 합니다.
        3. 추측에 기반하거나 불확실한 정보로 선지나 해설을 만들어서는 안 됩니다.
        **4. 반드시 모든 질문은 질문 형태로 끝나야합니다. 평서문, 감탄문과 같이 끝나면 안됩니다.

        [추가된 규칙: 오답 생성 전략]
        - 오답 선지는 아래 전략을 최소 1개 이상 사용하여 생성합니다.
          - 전략 1: 실제 법 조항의 핵심 단어나 숫자(기간, 금액 등)를 미묘하게 바꾸기
          - 전략 2: 법률 개정 전의 낡은 정보나 폐기된 판례를 활용하기
          - 전략 3: 유사하지만 다른 법률(예: '개인정보 보호법'과 '신용정보법')의 내용을 교묘하게 섞기

        [금지 사항 (Negative Constraints)]
        - **(중요) 법 조항 번호(예: 제29조, 제10조)를 직접적으로 묻거나 해설에 사용하지 마세요.** LLM이 만들어낼 확률이 높습니다. 대신 법의 '원칙'이나 '개념'을 설명하세요.
        - '다음 중 성격이 다른 하나는?' 과 같이 기준이 모호한 질문은 생성하지 마세요.
        - 단순히 법 조항 텍스트를 복사-붙여넣기 한 것처럼 보이는 선지는 만들지 마세요.

        [추가된 규칙: 해설 작성]
        - 'explanation' 항목에는 정답의 근거(법 조항, 판례 번호 등)를 명확히 제시해야 합니다.
        - 정답 해설뿐만 아니라, 나머지 오답 선지들이 왜 틀렸는지에 대한 간략한 설명도 포함해야 합니다.

        [최종 자기 검증 단계 (Self-Correction)]
        - 아래 JSON을 생성한 후, 최종 제출 전에 다음 항목을 스스로 검토하고, 만약 하나라도 문제가 있다면 JSON 내용을 수정하여 완벽하게 만드세요.
        - **검증 체크리스트:**
            1.  **정답의 유일성:** 정답 외에 다른 선지가 정답으로 해석될 여지는 없는가?
            2.  **사실의 정확성:** 질문, 정답, 오답, 해설에 최신 법규와 다른 내용은 없는가? 추측성 정보는 없는가?
            3.  **논리적 일관성:** 해설 내용이 질문 및 정답과 모순되지는 않는가? (예: 정답이 'A'인데, 해설에서 'A는 틀렸다'고 설명하는 경우)


        [세분화된 JSON 형식]
        - 반드시 아래 [JSON 형식]을 엄격하게 준수하여 답변해야 합니다. 코드 블록(` ```json ... ``` `)으로 감싸서 출력해주세요.
        {{
            "topic": "{topic}",
            "question": "생성된 질문 내용",
            "options": {{
                "1": "첫 번째 선지",
                "2": "두 번째 선지",
                "3": "세 번째 선지",
                "4": "네 번째 선지"
            }},
            "answer": "정답 선지의 번호 (예: '1')",
            "explanation": {{
                "correct_reason": "정답이 맞는 이유에 대한 상세한 해설 (관련 법 조항 명시 필수)",
                "incorrect_reasons": {{
                    "1": "1번 선지가 오답인 이유",
                    "2": "2번 선지가 오답인 이유",
                    "3": "3번 선지가 오답인 이유"
                }}
            }}
        }}

        ---
        [실제 생성 요청]
        [주제]: {topic}
        [요구사항]:
    """

def generate_credit_law_quiz(model, tokenizer, topic: str):
    """
    주어진 주제에 대해 관련 법률 객관식 문제를 생성합니다.
    """
    # 프롬프트 템플릿에 주제를 삽입하여 최종 프롬프트 완성
    prompt = get_prompt_template().format(topic=topic)
    
    # 모델 입력 형식에 맞게 변환 (채팅 형식)
    messages = [
        {"role": "system", "content": "당신은 대한민국 금융 및 정보보호 법률 분야의 최고 전문가입니다."},
        {"role": "user", "content": prompt}
    ]
    
    # 토크나이저를 사용하여 입력 텍스트를 모델이 이해할 수 있는 텐서로 변환
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 모델로부터 답변 생성
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,      # 질문, 선지, 해설을 충분히 생성할 수 있도록 설정
        do_sample=True,           # 다양한 결과를 위해 샘플링 사용
        temperature=0.2,          # 생성 결과의 창의성 조절 (낮을수록 결정적)
        top_p=0.9,                # 높은 확률의 단어 위주로 샘플링
        repetition_penalty=1.1,   # 반복적인 표현을 줄이기 위한 패널티
        eos_token_id=tokenizer.eos_token_id # 문장 끝 토큰 ID
    )
    
    # 생성된 결과에서 프롬프트 부분을 제외하고 디코딩
    response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # 생성된 텍스트에서 JSON 부분만 추출하여 파싱
    try:
        # 모델이 생성한 텍스트에서 ```json ... ``` 코드 블록을 찾아 추출
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        quiz_data = json.loads(json_str)
        # 생성된 데이터에 주제 정보 추가
        quiz_data['topic'] = topic
        return quiz_data
    except (IndexError, json.JSONDecodeError) as e:
        # JSON 파싱에 실패할 경우 오류 메시지와 원본 출력을 보여줌
        print(f"❌ 오류: 모델이 생성한 결과에서 유효한 JSON을 파싱하는 데 실패했습니다. (오류: {e})")
        print("\n--- 모델 원본 출력 ---")
        print(response_text)
        print("---------------------\n")
        return None

def main():
    """
    메인 실행 함수
    """
    model, tokenizer = setup_model()
    
    if not model or not tokenizer:
        return

    # 문제를 생성하고 싶은 주제 목록
    topics_to_generate = [
        "정보보호", "금융보안"
        # "정보보호", "금융보안", "신용정보법", "개인정보 보호법", 
        # "정보통신망법", "컴플라이언스", "금융 분야 클라우드", "암호화폐",
        # "APT공격", "보이스피싱", "지급결제시스템"
    ]
    
    # 각 주제별로 생성할 데이터 개수
    num_quizzes_per_topic = 5
    
    all_generated_quizzes = []
    
    for topic in topics_to_generate:
        print(f"\n{'='*20} 주제: '{topic}' (총 {num_quizzes_per_topic}개 생성) {'='*20}")
        topic_quizzes = []
        for i in range(num_quizzes_per_topic):
            print(f"▶ {topic} - 문제 {i+1}/{num_quizzes_per_topic} 생성 중...")
            quiz = generate_credit_law_quiz(model, tokenizer, topic)
            if quiz:
                topic_quizzes.append(quiz)
                print(f"✅ {topic} - 문제 {i+1} 생성 완료.")
        
        all_generated_quizzes.extend(topic_quizzes)
        print(f"\n🎉 주제 '{topic}'에 대한 문제 {len(topic_quizzes)}개 생성을 완료했습니다.")

    # 최종적으로 생성된 모든 문제를 하나의 JSON으로 파일에 저장
    if all_generated_quizzes:
        output_filename = "generated_law_quizzes_all.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_generated_quizzes, f, ensure_ascii=False, indent=2)
        print(f"\n\n🚀 총 {len(all_generated_quizzes)}개의 문제가 '{output_filename}' 파일에 성공적으로 저장되었습니다.")


if __name__ == "__main__":
    main()
