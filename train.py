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
DATASET_PATH = "/workspace/2025-AI-Challeng-finance/generated_dataset_midm.jsonl" 
OUTPUT_DIR = "./midm-lora-adapter-trainer"

# --- 2. 데이터셋 전처리 ---

# 토크나이저를 전처리 함수 밖에서 미리 로딩
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ★★★★★ 수정된 부분 ★★★★★
def preprocess_function(examples):
    prompt_template = """### 지시:
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
    options = examples['options']
    formatted_options = {
        'option_1': options.get('1', ''), 'option_2': options.get('2', ''),
        'option_3': options.get('3', ''), 'option_4': options.get('4', '')
    }
    prompt = prompt_template.format(question=examples['question'], **formatted_options)
    full_text = prompt + str(examples['answer'])
    
    # ★★★ padding=False -> padding="max_length"로 변경 ★★★
    tokenized_output = tokenizer(
        full_text, 
        truncation=True, 
        max_length=1024, 
        padding="max_length" # 모든 데이터를 최대 길이에 맞춰 패딩
    )
    
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    
    print("데이터셋 전처리를 시작합니다...")
    dataset = dataset.map(preprocess_function, remove_columns=list(dataset.features))
    
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

    # DataCollator는 이제 패딩을 따로 처리할 필요가 없지만, 배치를 만들어주기 위해 여전히 필요합니다.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("LoRA 파인튜닝을 시작합니다...")
    trainer.train()

    print(f"학습된 LoRA 어댑터를 '{OUTPUT_DIR}'에 저장합니다.")
    trainer.save_model(OUTPUT_DIR)

    print("파인튜닝이 완료되었습니다!")