import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

# --- 1. 설정 (Configuration) ---
PDF_PATH = "/workspace/2025-AI-Challeng-finance/pdf/개인정보 보호법(법률)(제19234호)(20250313).pdf"
OUTPUT_FILE = "/workspace/2025-AI-Challeng-finance/generated_dataset_midm.jsonl"
MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
CHUNK_SIZE = 1200
# ★★★★★ 청크당 생성할 문제 개수 설정 ★★★★★
GENERATIONS_PER_CHUNK = 10

# --- 2. 프롬프트 템플릿 (변경 없음) ---
PROMPT_STEP_1_QA = """당신은 주어진 문맥에서 핵심 질문과 그에 대한 정답을 추출하는 QA 봇입니다. 문맥을 바탕으로 질문 1개와 그에 대한 정확한 답변 1개를 생성하여 아래 JSON 형식으로 출력하세요. 다른 설명은 절대 추가하지 마십시오.

### 문맥:
{context}

### JSON 출력:
{{
  "question": "생성된 질문",
  "correct_answer": "생성된 정답"
}}
"""
PROMPT_STEP_2_DISTRACTORS = """당신은 매력적인 오답을 생성하는 AI입니다. 주어진 '문맥'과 '질문', 그리고 '정답'을 참고하여, 이 질문에 대한 그럴듯하지만 명백히 틀린 '오답' 3개를 생성하세요. 오답들은 서로 다른 내용을 다루어야 합니다. 최종 결과는 반드시 아래 JSON 형식으로만 출력해야 합니다. 다른 설명은 절대 추가하지 마십시오.

### 문맥:
{context}

### 질문:
{question}

### 정답:
{correct_answer}

### JSON 출력:
{{
  "distractors": [
    "첫 번째 오답",
    "두 번째 오답",
    "세 번째 오답"
  ]
}}
"""

# --- 3. 핵심 기능 함수 (변경 없음) ---
def pdf_to_chunks(pdf_path, chunk_size):
    print(f"'{pdf_path}'에서 텍스트를 추출 중...")
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
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
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
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

# --- 4. 메인 실행 로직 (★★★★★ 수정된 부분 ★★★★★) ---
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
            
            # 진행 상황 표시를 위해 총 생성 개수 계산
            total_generations = len(text_chunks) * GENERATIONS_PER_CHUNK
            print(f"\n총 {len(text_chunks)}개의 청크에 대해, 각 10회씩 총 {total_generations}개의 QA 생성을 시작합니다...")

            # tqdm을 수동으로 제어
            with tqdm(total=total_generations, desc="QA 생성 중") as pbar:
                for chunk in text_chunks:
                    if len(chunk) < 200 or "목차" in chunk or "제1조(목적)" in chunk or "부 칙" in chunk:
                        pbar.update(GENERATIONS_PER_CHUNK) # 건너뛰는 만큼 progress bar 업데이트
                        continue
                    
                    # ★★★ 하나의 청크에 대해 10번 반복 생성 ★★★
                    for i in range(GENERATIONS_PER_CHUNK):
                        # 1단계: 질문과 정답 생성
                        qa_prompt = PROMPT_STEP_1_QA.format(context=chunk)
                        qa_text = generate_with_local_model(model, tokenizer, qa_prompt)
                        qa_data = clean_and_parse_json(qa_text)

                        if not qa_data or 'question' not in qa_data or 'correct_answer' not in qa_data:
                            print(f"\n[실패] 1단계: 청크 #{text_chunks.index(chunk)+1}-{i+1} 질문/정답 생성 실패. 건너뜁니다.")
                            pbar.update(1)
                            continue

                        question = qa_data['question']
                        correct_answer = qa_data['correct_answer']
                        
                        # 2단계: 오답 생성
                        distractor_prompt = PROMPT_STEP_2_DISTRACTORS.format(context=chunk, question=question, correct_answer=correct_answer)
                        distractor_text = generate_with_local_model(model, tokenizer, distractor_prompt)
                        distractor_data = clean_and_parse_json(distractor_text)

                        if not distractor_data or 'distractors' not in distractor_data or len(distractor_data['distractors']) < 3:
                            print(f"\n[실패] 2단계: 청크 #{text_chunks.index(chunk)+1}-{i+1} 오답 생성 실패. 건너뜁니다.")
                            pbar.update(1)
                            continue
                        
                        # 3단계: 최종 JSON 조립
                        options_list = distractor_data['distractors'][:3] + [correct_answer]
                        random.shuffle(options_list)

                        final_json = {
                            "type": "multiple_choice",
                            "question": question,
                            "options": {str(i+1): option for i, option in enumerate(options_list)},
                            "answer": str(options_list.index(correct_answer) + 1),
                            "source_chunk": chunk
                        }
                        
                        f.write(json.dumps(final_json, ensure_ascii=False) + '\n')
                        f.flush()
                        generated_count += 1
                        pbar.update(1) # progress bar 1칸 업데이트
                        
        print(f"\n총 {generated_count}개의 QA 데이터가 생성되어 '{OUTPUT_FILE}' 파일에 저장되었습니다.")
        print("데이터 생성이 완료되었습니다!")