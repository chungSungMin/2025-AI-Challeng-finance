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

MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ 여기가 수정된 부분입니다 ★★★
# ★★★ 여러 개의 jsonl 파일 경로를 리스트로 지정 ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
DATASET_FILES = [
    "/workspace/open_qa.jsonl",
    "/workspace/open_qa1.jsonl",
    "/workspace/open_qa2.jsonl",
    "/workspace/2025-AI-Challeng-finance/cybersecurity_data_final_processed.jsonl",
    "/workspace/2025-AI-Challeng-finance/cybersecurity_data_translated_ko_nllb_from_5000.jsonl"
] 
OUTPUT_DIR = "./midm-lora-adapter-trainer"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_model_code=True)
if tokenizer.pad_token is None : 
    tokenizer.pad_token = tokenizer.eos_token


def preprocess_function_with_loss_mask_text(examples):
    # 프롬프트 템플릿 정의
    prompt_template = """
        다음 질문에 대한 옳은 답변을 선택지에서 고르세요.

        ### 질문 : 
        {question}

        ### 답변 : 
    """
    
    prompt = prompt_template.format(question=examples['question'])
    answer = str(examples['answer'])
    prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])

    full_text = prompt + answer
    tokenized_full = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,         # max_length를 넘으면 잘라냄
        max_length=1024,         # 최대 길이 설정
        padding='max_length'     # max_length에 맞춰 패딩 추가
    )
    
    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()
    
    labels[:min(prompt_len, 1024)] = [-100] * min(prompt_len, 1024) 

    labels = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels]

    tokenized_output = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": tokenized_full["attention_mask"],
        "original_len" : len(input_ids)
    }
    
    return tokenized_output

    

if __name__ == "__main__" : 
    # ★★★ 여러 파일을 한 번에 로드 ★★★
    dataset = load_dataset('json', data_files=DATASET_FILES, split='train')

    dataset = dataset.map(preprocess_function_with_loss_mask_text, remove_columns=list(dataset.features))

    ############ token 길이가 1024가 넘는 경우 제거 #############
    def filter_safe_data(example):
        return 0 < len(example['input_ids']) <= 1024

    safe_dataset = dataset.filter(filter_safe_data)
    final_dataset = safe_dataset.remove_columns(['original_len'])
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Final dataset size: {len(final_dataset)}")
    ############# ############# ############# ############# ############# #############


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, # 4비트 양자화
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", # 가중치 분포 최적화
        bnb_4bit_compute_dtype=torch.bfloat16 # 계산시 4비트를 16비트로 변환 -> 학습의 안정성
    )


    # AutoModelForCausalLM 은 다음에 올 단어를 예측하는 모델들이다. 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model) # 양자화 모델의 안정성을 높혀주는 함수


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16, # 주로 rank의 2배를 ㅏㅅ용하는 경향이 있다.
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        # num_train_epochs=3,      # Epoch 대신 Step으로 학습하기 위해 이 부분을 주석 처리하거나 삭제합니다.
        max_steps=2500,           # ★★★ 총 500 스텝만큼만 학습하도록 설정합니다. ★★★
        logging_steps=10,
        fp16=True,
        # save_strategy="epoch",   # 저장 전략을 step 단위로 변경합니다.
        save_strategy="steps",     # ★★★ "steps"로 변경 ★★★
        save_steps=500,            # ★★★ 100 스텝마다 모델을 저장하도록 설정합니다. ★★★
        optim="paged_adamw_8bit"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=final_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print('LoRA 파인튜닝 시작...')
    trainer.train()

    print(f"학습된 LoRA 어뎁터를 {OUTPUT_DIR}에 저장합니다. ")
    trainer.save_model(OUTPUT_DIR)

    print('Fine Tunning 종료..')