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
DATASET_PATH = "/workspace/merged_dataset.jsonl" 
OUTPUT_DIR = "./midm-lora-adapter-trainer"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_model_code=True)
if tokenizer.pad_token is None : 
    tokenizer.pad_token = tokenizer.eos_token



# def preprocess_function_multichoice(examples) : 
#     # prompt 템플릿 설정
#     prompt_template = """
#         다음 질문에 대한 옳은 답변을 선택지에서 고르세요.

#         ### 질문 : 
#         {question}

#         ### 선택지 : 
#         1. {option_1}
#         2. {option_2}
#         3. {option_3}
#         4. {option_4}

#         ### 답변 : 
#     """

#     # examples 라는 데이터셋에서 option과 question, answer를 추출한다.
#     options = examples['options']
#     formatted_options = {
#         'option_1' : options.get('1',''), 'option_2' : options.get('2',''),
#         'option_3' : options.get('3', ''), 'option_4' : options.get('4', '')
#     }

#     # **formatted_options 여기서 '**'는 dictionary unpacking을 의미한다.
#     prompt = prompt_template.format(question=examples['question'], **formatted_options)
#     full_text = prompt + str(examples['answer'])

#     tokenized_output = tokenizer(
#         full_text,
#         truncation=True,
#         max_length=1024,
#         # padding="longest" 를 사용하면 배치 내 가장 긴 토큰으로 맞춰지게된다.
#         padding='max_length'
#     )
    
#     """tokenized_output
#     해당 변순의 경우 크게 2가지로 구성되어있습니다.
#         input_ids : []  # 입력 prompt의 token들의 리스트
#         attention_maks : [] # 실제 입력과 패딩을 구별하기 위한 리스트
#     """


#     tokenized_output["label"] = tokenized_output["input_ids"].copy()
#     return tokenized_output


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
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
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
        num_train_epochs=5,
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
