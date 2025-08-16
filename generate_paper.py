import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import re

# --- 1. 설정 (Configuration) ---
PDF_PATH = "/workspace/2025-AI-Challeng-finance/data/정보보호산업의 진흥에 관한 법률(법률)(제19990호)(20240710).pdf"
OUTPUT_FILE = "/workspace/2025-AI-Challeng-finance/generated_finetuning_dataset_from_paper.jsonl"
MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
CHUNK_SIZE = 1200 # 논문과 유사하게 충분한 컨텍스트를 제공하도록 설정

# 후처리를 위한 설정
SIMILARITY_MODEL_ID = 'jhgan/ko-sroberta-multitask' # 한국어 의미 비교 모델
SIMILARITY_THRESHOLD_UNANSWERED = 0.9 # '답변 불가'와 유사도 임계값
SIMILARITY_THRESHOLD_UNRELATED = 0.6 # 질문-답변, 질문-문맥 간 관련성 임계값

# --- 2. 논문 기반 프롬프트 템플릿 ---
PROMPT_QUESTION_GENERATION = """주어진 문맥과 관련하여 가장 흥미롭고 사실에 입각한 질문을 최대 10개까지 생성하세요. 질문은 반드시 리스트 형식으로 반환해야 합니다. 다른 설명은 절대 추가하지 마십시오.

### 문맥:
{context}

### 질문 리스트:
"""

PROMPT_ANSWER_GENERATION = """주어진 문맥을 바탕으로 다음 질문에 대해 최대한 사실적으로 답변하세요. 주어진 내용으로 사실적인 답변이 불가능할 경우, "주어진 내용 기반으로는 사실적인 답변이 불가능합니다."라고만 응답하세요.

### 문맥:
{context}

### 질문:
{question}

### 답변:
"""

# --- 3. 핵심 기능 함수 (후처리 함수 추가) ---
def pdf_to_chunks(pdf_path, chunk_size):
    print(f"'{pdf_path}'에서 텍스트를 추출 중...")
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        # 공백 및 줄바꿈 정제
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
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response_text.strip()

# ★★★★★ 논문 방법론 기반 후처리 함수들 ★★★★★

def check_unanswered(answer, similarity_model):
    """미답변 QA 확인: 답변이 '답변 불가' 메시지와 유사한지 확인"""
    no_answer_phrase = "주어진 내용 기반으로는 사실적인 답변이 불가능합니다."
    embedding_answer = similarity_model.encode(answer, convert_to_tensor=True)
    embedding_no_answer = similarity_model.encode(no_answer_phrase, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding_answer, embedding_no_answer)[0][0].item()
    return cosine_score > SIMILARITY_THRESHOLD_UNANSWERED

def check_unfinished(answer):
    """미완성 QA 확인: 답변의 마지막 문장이 제대로 끝나는지 확인 (간단한 버전)"""
    # 논문은 RoBERTa 기반 분류 모델을 사용했지만, 여기서는 간단한 휴리스틱으로 구현
    # 답변이 비어있거나, 문장의 끝이 ., ?, ! 와 같은 문장 부호로 끝나지 않으면 미완성으로 간주
    if not answer or not answer.strip()[-1] in ['.', '?', '!', '다', ')']:
        return True
    return False

def check_unrelated(question, answer, context, similarity_model):
    """무관 QA 쌍 확인: 질문-답변, 질문-문맥 간의 관련성 확인"""
    # 질문과 답변의 관련성 점수
    q_embedding = similarity_model.encode(question, convert_to_tensor=True)
    a_embedding = similarity_model.encode(answer, convert_to_tensor=True)
    qa_score = util.pytorch_cos_sim(q_embedding, a_embedding)[0][0].item()

    # 질문과 문맥의 관련성 점수
    c_embedding = similarity_model.encode(context, convert_to_tensor=True)
    qc_score = util.pytorch_cos_sim(q_embedding, c_embedding)[0][0].item()

    # 둘 중 하나라도 관련성이 낮으면 무관한 쌍으로 판단
    if qa_score < SIMILARITY_THRESHOLD_UNRELATED or qc_score < SIMILARITY_THRESHOLD_UNRELATED:
        return True
    return False

# --- 4. 메인 실행 로직 (논문 방법론 적용) ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다. GPU 환경을 확인하세요.")

    # LLM 로딩
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"'{MODEL_ID}' 모델과 토크나이저를 로딩합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")

    # 의미 유사도 모델 로딩
    print(f"'{SIMILARITY_MODEL_ID}' 의미 유사도 모델을 로딩합니다...")
    similarity_model = SentenceTransformer(SIMILARITY_MODEL_ID, device='cuda')

    text_chunks = pdf_to_chunks(PDF_PATH, CHUNK_SIZE)

    if text_chunks:
        final_qa_pairs = []
        # tqdm을 청크 단위로 설정
        for chunk in tqdm(text_chunks, desc="청크 처리 중"):
            if len(chunk.strip()) < 200: continue # 너무 짧은 청크는 건너뛰기

            # 1단계: 청크에서 질문 목록 생성
            qg_prompt = PROMPT_QUESTION_GENERATION.format(context=chunk)
            questions_text = generate_with_local_model(model, tokenizer, qg_prompt)
            
            # 생성된 텍스트에서 질문들만 파싱 (e.g., "1. 질문내용\n2. 질문내용")
            questions = [q.strip() for q in re.split(r'\d+\.\s*', questions_text) if q.strip()]
            if not questions:
                continue

            # 2단계: 각 질문에 대해 답변 생성 및 후처리
            for question in questions:
                # 답변 생성
                ag_prompt = PROMPT_ANSWER_GENERATION.format(context=chunk, question=question)
                answer = generate_with_local_model(model, tokenizer, ag_prompt)

                # 3단계: 후처리 (품질 검증)
                if check_unanswered(answer, similarity_model):
                    # print(f"\n[정제] 미답변 QA 쌍 제거: {question}")
                    continue
                if check_unfinished(answer):
                    # print(f"\n[정제] 미완성 QA 쌍 제거: {answer}")
                    continue
                if check_unrelated(question, answer, chunk, similarity_model):
                    # print(f"\n[정제] 무관 QA 쌍 제거: {question}")
                    continue
                
                # 모든 검증을 통과한 QA 쌍만 최종 리스트에 추가
                final_qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "source_chunk": chunk
                })

        # 4단계: 최종 결과를 파일에 저장
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for qa_pair in final_qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

        print(f"\n총 {len(final_qa_pairs)}개의 고품질 QA 데이터가 생성되어 '{OUTPUT_FILE}' 파일에 저장되었습니다.")
        print("데이터 생성이 완료되었습니다!")