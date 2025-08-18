import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import re

# --- 1. 설정 (Configuration) ---

# ⭐️ 수정 1: 여러 PDF 파일 경로를 리스트로 관리
PDF_PATHS = [
    "/workspace/2025-AI-Challeng-finance/data/정보보호산업의 진흥에 관한 법률(법률)(제19990호)(20240710).pdf",
    # 여기에 다른 PDF 파일 경로를 추가하세요.
    # "/path/to/your/second.pdf",
    # "/path/to/your/third.pdf"
]
OUTPUT_FILE = "/workspace/2025-AI-Challeng-finance/generated_finetuning_dataset_from_multiple_pdfs.jsonl"

# ⭐️ 수정 2: 더 성능이 좋은 최신 모델로 변경 (추천)
MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"
SIMILARITY_MODEL_ID = 'BAAI/bge-m3' # 한국어 성능이 매우 뛰어난 최신 임베딩 모델

CHUNK_SIZE = 1200

# 후처리 임계값 설정
SIMILARITY_THRESHOLD_UNANSWERED = 0.9
SIMILARITY_THRESHOLD_UNRELATED = 0.6

# --- 2. 프롬프트 템플릿 (SOLAR 모델에 최적화) ---
PROMPT_QUESTION_GENERATION = """### User:
주어진 문맥과 관련하여, 사실에 입각한 중요한 질문을 최대 10개까지 생성해줘. 질문은 반드시 번호가 매겨진 리스트 형식으로만 반환해야 해. 다른 설명은 절대 추가하지 마.

### 문맥:
{context}

### Assistant:
"""

PROMPT_ANSWER_GENERATION = """### User:
주어진 문맥을 바탕으로 다음 질문에 대해 사실적으로 답변해줘. 주어진 내용으로 답변이 불가능하면, "주어진 내용 기반으로는 사실적인 답변이 불가능합니다."라고만 응답해야 해.

### 문맥:
{context}

### 질문:
{question}

### Assistant:
"""

# --- 3. 핵심 기능 함수 ---
def pdf_to_chunks(pdf_path, chunk_size):
    print(f"\n'{os.path.basename(pdf_path)}'에서 텍스트를 추출 중...")
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

# ⭐️ 수정 3: 모델 생성 로직 단순화 및 개선
def generate_with_local_model(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response_text.strip()

# --- 후처리 함수들 (기존과 동일) ---
def check_unanswered(answer, similarity_model):
    no_answer_phrase = "주어진 내용 기반으로는 사실적인 답변이 불가능합니다."
    embedding_answer = similarity_model.encode(answer, convert_to_tensor=True)
    embedding_no_answer = similarity_model.encode(no_answer_phrase, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding_answer, embedding_no_answer)[0][0].item()
    return cosine_score > SIMILARITY_THRESHOLD_UNANSWERED

def check_unfinished(answer):
    if not answer or not answer.strip()[-1] in ['.', '?', '!', '다', ')', '"', "'"]:
        return True
    return False

def check_unrelated(question, answer, context, similarity_model):
    q_embedding = similarity_model.encode(question, convert_to_tensor=True)
    a_embedding = similarity_model.encode(answer, convert_to_tensor=True)
    qa_score = util.pytorch_cos_sim(q_embedding, a_embedding)[0][0].item()

    c_embedding = similarity_model.encode(context, convert_to_tensor=True)
    qc_score = util.pytorch_cos_sim(q_embedding, c_embedding)[0][0].item()

    if qa_score < SIMILARITY_THRESHOLD_UNRELATED or qc_score < SIMILARITY_THRESHOLD_UNRELATED:
        return True
    return False

# --- 4. 메인 실행 로직 (여러 파일 처리) ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다. GPU 환경을 확인하세요.")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"'{MODEL_ID}' 모델과 토크나이저를 로딩합니다...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")

    print(f"'{SIMILARITY_MODEL_ID}' 의미 유사도 모델을 로딩합니다...")
    similarity_model = SentenceTransformer(SIMILARITY_MODEL_ID, device='cuda')

    # ⭐️ 수정 4: 모든 PDF의 QA 쌍을 저장할 리스트
    final_qa_pairs = []

    # ⭐️ 수정 5: PDF 파일 리스트를 순회하는 루프 추가
    for pdf_path in PDF_PATHS:
        text_chunks = pdf_to_chunks(pdf_path, CHUNK_SIZE)

        if not text_chunks:
            continue

        for chunk in tqdm(text_chunks, desc=f"'{os.path.basename(pdf_path)}' 청크 처리 중"):
            if len(chunk.strip()) < 200: continue

            qg_prompt = PROMPT_QUESTION_GENERATION.format(context=chunk)
            questions_text = generate_with_local_model(model, tokenizer, qg_prompt)
            
            questions = [q.strip() for q in re.split(r'\d+\.\s*', questions_text) if q.strip()]
            if not questions:
                continue

            for question in questions:
                ag_prompt = PROMPT_ANSWER_GENERATION.format(context=chunk, question=question)
                answer = generate_with_local_model(model, tokenizer, ag_prompt)

                if check_unanswered(answer, similarity_model) or \
                   check_unfinished(answer) or \
                   check_unrelated(question, answer, chunk, similarity_model):
                    continue
                
                final_qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "source_chunk": chunk
                })

    # 모든 처리가 끝난 후 파일에 한 번에 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for qa_pair in final_qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

    print(f"\n총 {len(final_qa_pairs)}개의 고품질 QA 데이터가 생성되어 '{OUTPUT_FILE}' 파일에 저장되었습니다.")
    print("모든 PDF 파일 처리가 완료되었습니다!")
