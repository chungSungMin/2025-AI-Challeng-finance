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

# ⭐️ 수정 1: 기본 모델 ID를 SOLAR로 변경
MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"
DATASET_PATH = "/workspace/merged_dataset.jsonl" 
# ⭐️ 수정 2: 출력 디렉터리 이름 변경
OUTPUT_DIR = "./solar-lora-adapter-trainer"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None : 
    tokenizer.pad_token = tokenizer.eos_token

# ⭐️ 수정 3: SOLAR 모델에 최적화된 프롬프트 템플릿으로 전처리 함수 변경
def preprocess_function_with_loss_mask_text(examples):
    """
    SOLAR 모델의 공식 프롬프트 템플릿에 맞춰 데이터를 전처리하고,
    답변 부분만 학습하도록 loss mask를 적용하는 함수입니다.
    """
    # SOLAR 프롬프트 템플릿의 User 부분 구성
    instruction = f"다음 질문에 대한 옳은 답변을 선택지에서 고르세요.\n\n### 질문 :\n{examples['question']}"
    user_prompt = f"### User:\n{instruction}\n\n### Assistant:\n"
    
    answer = str(examples['answer'])

    # User 프롬프트 부분의 길이를 정확히 계산 (이 부분은 학습에서 제외)
    prompt_len = len(tokenizer(user_prompt, add_special_tokens=True)["input_ids"])

    # 전체 텍스트(User + Assistant)를 합쳐서 토큰화
    full_text = user_prompt + answer
    tokenized_full = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=1024,
        padding='max_length'
    )
    
    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()
    
    # User 프롬프트 부분은 loss 계산에서 제외 (-100으로 마스킹)
    labels[:min(prompt_len, 1024)] = [-100] * min(prompt_len, 1024)

    # 패딩 토큰도 loss 계산에서 제외
    labels = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels]

    tokenized_output = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": tokenized_full["attention_mask"],
    }
    
    return tokenized_output

if __name__ == "__main__" : 
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

    # ⭐️ 수정 4: 전처리 함수에서 패딩/잘라내기를 모두 처리하므로, 후속 필터링 로직은 제거합니다.
    final_dataset = dataset.map(preprocess_function_with_loss_mask_text, remove_columns=list(dataset.features))
    
    print(f"Processed dataset size: {len(final_dataset)}")

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
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        # SOLAR 모델의 어텐션 모듈 이름에 맞게 target_modules 수정
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=10,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
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
