import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 1. 설정 (Configuration) ---
PDF_PATH = "/workspace/2025-AI-Challeng-finance/pdf/개인정보 보호법(법률)(제19234호)(20250313).pdf"
OUTPUT_FILE = "/workspace/2025-AI-Challeng-finance/generated_dataset_midm_대통령_keyword_v2.jsonl"
MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
CHUNK_SIZE = 1200
GENERATIONS_PER_CHUNK = 10

# --- 2. 프롬프트 템플릿 (★★★★★ 완료 조건 강화 ★★★★★) ---
PROMPT_KEYWORD_QA = """당신은 주어진 문맥에서 중요한 '키워드'를 중심으로 하는 주관식 문제를 생성하는 AI입니다.

다음 단계를 따르세요:
1. '문맥'에서 핵심적인 키워드를 기반으로 2~3줄로 핵심을 담은 완전한 정답 문장을 만들어주세요.
2. 이 키워드가 정답이 되는 질문을 생성합니다. 질문은 빈칸 채우기 형식이거나, "~란 무엇인가?", "~을 서술하시오" 와 같은 형태가 좋습니다.
3. 다른 부가적인 설명은 절대 추가하지 말고, 반드시 아래 JSON 형식으로만 결과를 출력하세요.
4. 정답 문장은 반드시 마침표(.)로 끝나야 합니다.

### 문맥:
{context}

### JSON 출력:
{{
  "question": "생성된 질문",
  "answer": "정답이 되는 핵심 키워드 기반의 완전한 문장."
}}
"""

# --- 3. 핵심 기능 함수 ---
def pdf_to_chunks(pdf_path, chunk_size):
    print(f"'{pdf_path}'에서 텍스트를 추출 중...")
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        print(f"총 {len(chunks)}개의 청크로 분할 완료.")
        return chunks
    except Exception as e:
        print(f"PDF 처리 오류: {e}")
        return []

def generate_with_local_model(model, tokenizer, prompt):
    conversation = [{'role': 'user', 'content': prompt}]
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    # ★★★★★ max_new_tokens 값을 1024로 늘려서 생성 공간 확보 ★★★★★
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=1024, # 충분한 공간 제공
        do_sample=True, 
        temperature=0.6, 
        top_p=0.9, 
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response_text

def clean_and_parse_json(text):
    try:
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            json_str = text[start_index:end_index]
            return json.loads(json_str)
        return None
    except Exception as e:
        print(f"\n[경고] JSON 파싱 실패. 모델 원본 텍스트: '{text}'. 오류: {e}")
        return None

# --- 4. 메인 실행 로직 ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다. GPU 환경을 확인하세요.")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    print(f"'{MODEL_ID}' 모델과 토크나이저를 로딩합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")

    text_chunks = pdf_to_chunks(PDF_PATH, CHUNK_SIZE)

    if text_chunks:
        generated_count = 0
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            total_generations = len(text_chunks) * GENERATIONS_PER_CHUNK
            print(f"\n총 {len(text_chunks)}개의 청크에 대해, 각 {GENERATIONS_PER_CHUNK}회씩 총 {total_generations}개의 키워드 주관식 QA 생성을 시작합니다...")

            with tqdm(total=total_generations, desc="키워드 QA 생성 중") as pbar:
                for chunk in text_chunks:
                    if len(chunk) < 150 or "목차" in chunk or "제1조(목적)" in chunk or "부 칙" in chunk:
                        pbar.update(GENERATIONS_PER_CHUNK)
                        continue

                    for i in range(GENERATIONS_PER_CHUNK):
                        prompt = PROMPT_KEYWORD_QA.format(context=chunk)
                        generated_text = generate_with_local_model(model, tokenizer, prompt)
                        qa_data = clean_and_parse_json(generated_text)

                        if not qa_data or 'question' not in qa_data or 'answer' not in qa_data:
                            print(f"\n[실패] 청크 #{text_chunks.index(chunk)+1}-{i+1} QA 생성 실패. 건너뜁니다.")
                            pbar.update(1)
                            continue
                        
                        final_json = {
                            "type": "short_answer_sentence", # 타입 이름 변경
                            "question": qa_data['question'],
                            "answer": qa_data['answer'],
                            "source_chunk": chunk
                        }

                        f.write(json.dumps(final_json, ensure_ascii=False) + '\n')
                        f.flush()
                        generated_count += 1
                        pbar.update(1)

        print(f"\n총 {generated_count}개의 키워드 주관식 QA 데이터가 생성되어 '{OUTPUT_FILE}' 파일에 저장되었습니다.")
        print("데이터 생성이 완료되었습니다!")
