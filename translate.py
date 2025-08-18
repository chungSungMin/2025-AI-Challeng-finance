import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# --- 설정 ---
# ★★★ 모델 ID를 NLLB 모델로 변경 ★★★
MODEL_ID = "facebook/nllb-200-distilled-600M"
INPUT_FILE = '/workspace/2025-AI-Challeng-finance/cybersecurity_data_converted.jsonl'
# 출력 파일 이름도 모델에 맞게 변경하여 혼동 방지
OUTPUT_FILE = 'cybersecurity_data_translated_ko_nllb.jsonl'
BATCH_SIZE = 32
# --------------------------------------------------

def count_lines(filename):
    """파일의 총 줄 수를 세는 함수"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for line in f)

def main():
    # 1. 모델 및 토크나이저 로딩
    print(f"'{MODEL_ID}' 모델과 토크나이저를 로딩합니다...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ★★★ NLLB 모델은 소스 언어를 지정해야 합니다 ★★★
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"모델 로딩 완료. {device}를 사용하여 번역을 시작합니다.")

    # 진행률 표시를 위해 전체 줄 수 계산
    total_lines = count_lines(INPUT_FILE)

    # 입력 파일과 출력 파일을 미리 열어둠
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        batch_data = [] # 원본 JSON 데이터를 담을 리스트
        batch_texts = [] # 번역할 텍스트만 담을 리스트

        # tqdm으로 전체 파일 진행률을 표시
        for line in tqdm(infile, total=total_lines, desc="파일 처리 중"):
            data = json.loads(line)
            batch_data.append(data)
            batch_texts.append(data['question'])
            batch_texts.append(data['answer'])

            # 배치가 꽉 차면 즉시 번역하고 파일에 씀
            if len(batch_data) >= BATCH_SIZE:
                process_and_write_batch(batch_data, batch_texts, model, tokenizer, device, outfile)
                # 다음 배치를 위해 리스트 비우기
                batch_data.clear()
                batch_texts.clear()
        
        # 마지막에 남은 데이터 처리
        if batch_data:
            process_and_write_batch(batch_data, batch_texts, model, tokenizer, device, outfile)

    print("✅ 모든 작업이 성공적으로 완료되었습니다!")

def process_and_write_batch(batch_data, batch_texts, model, tokenizer, device, outfile):
    """배치를 번역하고 결과를 파일에 쓰는 함수"""
    with torch.no_grad():
        # 토크나이징
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # ★★★ NLLB 모델은 타겟 언어 ID를 generate 함수에 전달해야 합니다 ★★★
        # ★★★ 수정된 부분: tokenizer.convert_tokens_to_ids() 메서드를 사용 ★★★
        # 한국어 코드는 "kor_Hang" 입니다.
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"),
            max_length=512
        )
        translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # 번역된 내용을 다시 JSONL 파일로 저장
    translated_idx = 0
    for original_item in batch_data:
        new_item = {
            'question': translated_texts[translated_idx],
            'answer': translated_texts[translated_idx + 1]
        }
        outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        translated_idx += 2

if __name__ == "__main__":
    main()
