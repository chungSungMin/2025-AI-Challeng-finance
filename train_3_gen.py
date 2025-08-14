import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. 설정 (Configuration) ---
MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
OUTPUT_DIR = "./midm-lora-adapter-unified-trainer"

# 학습할 데이터 파일 리스트
SOURCE_DATA_FILES = [
    "/workspace/2025-AI-Challeng-finance/generated_dataset_midm_대통령_keyword.jsonl",
    "/workspace/2025-AI-Challeng-finance/generated_dataset_midm_대통령.jsonl",
    "/workspace/2025-AI-Challeng-finance/generated_dataset_x10.jsonl"
]
# 병합 및 정제된 데이터셋 파일 경로
COMBINED_DATASET_PATH = "/workspace/2025-AI-Challeng-finance/combined_training_dataset_cleaned.jsonl"

# --- 2. 데이터 파일 병합 및 정제 (★★★★★ 수정된 부분 ★★★★★) ---
def merge_and_clean_jsonl_files(file_list, output_file):
    """
    여러 개의 jsonl 파일을 하나로 합치면서, 
    객관식 선택지의 형식을 일관된 문자열로 통일합니다.
    """
    print(f"'{len(file_list)}'개의 데이터 파일을 병합하고 정제하여 '{output_file}'에 저장합니다.")
    
    def get_clean_option_text(option_value):
        """옵션 값이 dict 형태일 경우 content를 추출하고, 아니면 문자열로 변환합니다."""
        if isinstance(option_value, dict):
            return option_value.get('content', '')
        return str(option_value)

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for fname in file_list:
                print(f" - 처리 중: {fname}")
                with open(fname, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        try:
                            data = json.loads(line)
                            
                            # 객관식 문제의 'options' 필드 정제
                            if data.get('type') == 'multiple_choice' and 'options' in data and isinstance(data['options'], dict):
                                clean_options = {}
                                for key, value in data['options'].items():
                                    clean_options[key] = get_clean_option_text(value)
                                data['options'] = clean_options
                            
                            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                        except json.JSONDecodeError:
                            print(f"[경고] JSON 파싱 오류. 다음 라인을 건너뜁니다: {line.strip()}")
                            continue
        print("파일 병합 및 정제 완료!")
    except FileNotFoundError as e:
        print(f"[오류] 파일을 찾을 수 없습니다: {e.filename}")
        print("SOURCE_DATA_FILES 경로를 확인해주세요.")
        exit()


# --- 3. 데이터셋 전처리 (객관식/주관식 자동 감지) ---

# 프롬프트 템플릿 정의
MC_PROMPT_TEMPLATE = """### 지시:
다음 질문에 대한 올바른 답변을 선택지에서 고르시오.

### 질문:
{question}

### 선택지:
1. {option_1}
2. {option_2}
3. {option_3}
4. {option_4}

### 답변:
"""

SA_PROMPT_TEMPLATE = """### 지시:
다음 질문에 대해 서술하시오.

### 질문:
{question}

### 답변:
"""

# 토크나이저 미리 로딩
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(example):
    """데이터 타입에 따라 다른 프롬프트를 적용하는 전처리 함수"""
    
    question = example.get('question', '')
    answer = str(example.get('answer', ''))

    # 데이터 타입에 따라 프롬프트 선택
    if example.get('type') == 'multiple_choice' and 'options' in example:
        # 객관식 문제 처리
        options = example.get('options', {})
        formatted_options = {
            'option_1': str(options.get('1', '')),
            'option_2': str(options.get('2', '')),
            'option_3': str(options.get('3', '')),
            'option_4': str(options.get('4', '')),
        }
        prompt = MC_PROMPT_TEMPLATE.format(question=question, **formatted_options)
        full_text = prompt + answer
    else:
        # 주관식 문제 처리 (기본값)
        prompt = SA_PROMPT_TEMPLATE.format(question=question)
        full_text = prompt + answer

    # 토크나이징
    tokenized_output = tokenizer(
        full_text, 
        truncation=True, 
        max_length=1024, 
        padding="max_length"
    )
    
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

# --- 4. 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. 데이터 파일 병합 및 정제 실행
    merge_and_clean_jsonl_files(SOURCE_DATA_FILES, COMBINED_DATASET_PATH)

    # 2. 정제된 데이터셋 로드
    print(f"정제된 데이터셋 '{COMBINED_DATASET_PATH}'을 로딩합니다...")
    dataset = load_dataset('json', data_files=COMBINED_DATASET_PATH, split='train')
    
    # 3. 데이터셋 전처리
    print("데이터셋 전처리를 시작합니다 (객관식/주관식 자동 감지)...")
    dataset = dataset.map(preprocess_function, remove_columns=list(dataset.features))
    
    # 4. 모델 및 학습 설정 (기존과 동일)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"'{MODEL_ID}'의 베이스 모델을 로딩합니다...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    print("훈련 가능한 파라미터:")
    model.print_trainable_parameters()

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        optim="paged_adamw_8bit"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 5. 학습 시작
    print("LoRA 파인튜닝을 시작합니다...")
    trainer.train()

    print(f"학습된 LoRA 어댑터를 '{OUTPUT_DIR}'에 저장합니다.")
    trainer.save_model(OUTPUT_DIR)

    print("파인튜닝이 완료되었습니다!")
