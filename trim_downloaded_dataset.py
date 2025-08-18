import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# --- 설정 (사용자 환경에 맞게 수정) ---

# 데이터 정제 및 변환에 사용할 LLM
MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"

# 원본 파일 경로 (번역 결과 파일)
INPUT_FILE = '/workspace/2025-AI-Challeng-finance/cybersecurity_data_translated_ko_nllb.jsonl'
# 최종 결과를 저장할 새로운 파일 경로
OUTPUT_FILE = 'cybersecurity_data_final_processed.jsonl'

# -----------------------------------------

def is_abnormal(text: str) -> bool:
    """
    텍스트가 비정상적인 반복 패턴을 포함하는지 확인하는 함수.
    
    Args:
        text (str): 검사할 텍스트.

    Returns:
        bool: 비정상적이면 True, 그렇지 않으면 False.
    """
    if not text or len(text.split()) < 5:
        return True # 내용이 없거나 너무 짧으면 비정상으로 간주

    # 1. 단일 문자가 10번 이상 반복되는 경우 (예: "용용용...")
    if re.search(r'(.)\1{9,}', text):
        return True

    # 2. 동일한 단어가 5번 이상 연속으로 반복되는 경우 (예: "보안 보안 보안...")
    words = text.split()
    for i in range(len(words) - 4):
        if words[i] == words[i+1] == words[i+2] == words[i+3] == words[i+4]:
            return True
            
    # 3. 과도한 줄바꿈 문자열이 포함된 경우
    if "\\n\\n\\n\\n" in text:
        return True

    return False

def is_question(text: str) -> bool:
    """
    텍스트가 질문 형식인지 간단하게 확인하는 함수.
    """
    return text.strip().endswith('?')

def rephrase_as_question(text: str, pipe) -> str:
    """
    주어진 텍스트를 LLM을 사용하여 질문 형태로 변환하는 함수.
    """
    prompt = f"""### 지시:
다음 문장을 자연스러운 한국어 질문 형태로 바꿔주세요. 다른 설명 없이 질문 문장만 생성하세요.

### 문장:
{text}

### 질문:
"""
    try:
        output = pipe(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
            return_full_text=False,
            eos_token_id=pipe.tokenizer.eos_token_id
        )
        generated_text = output[0]['generated_text'].strip()
        
        # 생성된 텍스트 후처리
        if "###" in generated_text:
            generated_text = generated_text.split("###")[0].strip()
        
        # 마지막에 물음표가 없으면 추가
        if not generated_text.endswith('?'):
            generated_text += '?'
            
        return generated_text
    except Exception as e:
        print(f"질문 변환 중 오류 발생: {e}")
        return text # 오류 발생 시 원본 텍스트 반환

def main():
    """
    JSONL 파일을 읽어 데이터를 정제 및 변환하고 새 파일에 저장하는 메인 함수.
    """
    # 1. LLM 로딩
    print(f"'{MODEL_ID}' 모델과 토크나이저를 로딩합니다...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("모델 로딩 완료.")

    # 2. 데이터 처리
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            
            print(f"'{INPUT_FILE}' 파일의 데이터 처리을 시작합니다...")
            
            # 파일의 총 줄 수를 세어 tqdm에 사용
            num_lines = sum(1 for line in open(INPUT_FILE, 'r', encoding='utf-8'))
            infile.seek(0) # 파일 포인터를 다시 처음으로

            for line in tqdm(infile, total=num_lines, desc="데이터 처리 중"):
                data = json.loads(line)
                
                # 'answer' 필드가 비정상적이면 해당 데이터 라인을 건너뜀 (제거)
                if 'answer' not in data or is_abnormal(data['answer']):
                    continue
                
                # 'question' 필드가 질문이 아니면 질문으로 변환
                if 'question' in data and not is_question(data['question']):
                    data['question'] = rephrase_as_question(data['question'], pipe)
                
                # 처리된 데이터를 새 파일에 쓰기
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"✅ 데이터 처리 완료! 결과가 '{OUTPUT_FILE}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"❌ 오류: '{INPUT_FILE}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
