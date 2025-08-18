import torch
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# peft is no longer needed as we are not loading a LoRA adapter
# from peft import PeftModel 
import os
from datasets import load_dataset
from langchain_community.document_loaders import PyPDFLoader

# RAG libraries
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. Configuration ---

# ★★★ Changed Model ID to SOLAR ★★★
BASE_MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"

# --- LoRA adapter path is no longer needed ---
# LORA_ADAPTER_PATH = "/path/to/your/lora/adapter" 

# Test data and submission file paths
TEST_CSV_PATH = '/workspace/open/test.csv'
SUBMISSION_CSV_PATH = './submission_rag_solar_inference.csv' 

# List of PDF files for the RAG knowledge base
RAG_DATA_FILES = [
    "/workspace/개인금융채권의 관리 및 개인금융채무자의 보호에 관한 법률(법률)(제20369호)(20241017).pdf",
    "/workspace/개인정보 보호법(법률)(제19234호)(20250313) (1).pdf",
    # "/workspace/거버넌스.pdf",
    # "/workspace/경찰공무원 등의 개인정보 처리에 관한 규정(대통령령)(제35039호)(20241203).pdf",
    # "/workspace/금융보안연구원.pdf",
    # "/workspace/금융소비자 보호에 관한 법률(법률)(제20305호)(20240814).pdf",
    # "/workspace/금융실명거래 및 비밀보장에 관한 법률 시행규칙(총리령)(제01406호)(20170726).pdf",
    # "/workspace/랜섬웨어.pdf",
    # "/workspace/마이데이터.pdf",
    # "/workspace/메타버스.pdf",
    "/workspace/법원 개인정보 보호에 관한 규칙(대법원규칙)(제03109호)(20240315).pdf",
    # "/workspace/아웃소싱.pdf",
    # "/workspace/정보_보안.pdf",
    # "/workspace/클라우드컴퓨팅 발전 및 이용자 보호에 관한 법률(법률)(제20732호)(20250131).pdf",
]
# Embedding model and vector DB path configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
FAISS_DB_PATH = "./faiss_index_laws_solar"


# --- 2. RAG Backend Build Function ---
def build_or_load_rag_backend():
    """PDF 및 JSONL 파일들로부터 FAISS 벡터 DB를 구축하거나 기존 DB를 로드합니다."""
    
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if os.path.exists(FAISS_DB_PATH):
        print(f"⏳ 기존 벡터 DB를 '{FAISS_DB_PATH}'에서 로드합니다...")
        vector_db = FAISS.load_local(
            FAISS_DB_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ 벡터 DB 로드 완료.")
    else:
        print(f"⏳ '{RAG_DATA_FILES}' 파일들로 새로운 벡터 DB를 구축합니다...")
        
        all_documents = []
        for file_path in tqdm(RAG_DATA_FILES, desc="📚 PDF 파일 로딩 중"):
            try:
                loader = PyPDFLoader(file_path)
                documents_from_pdf = loader.load() 
                all_documents.extend(documents_from_pdf)
            except Exception as e:
                print(f"⚠️ 경고: '{file_path}' 파일을 처리하는 중 오류 발생: {e}")
        
        if not all_documents:
            print("❌ 오류: 처리할 문서가 없습니다. RAG_DATA_FILES 경로를 확인해주세요.")
            exit()
            
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★ 이 부분이 핵심입니다! 문서를 작은 단위로 분할 ★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=124)
        split_documents = text_splitter.split_documents(all_documents)
        print(f"✅ 문서를 총 {len(split_documents)}개의 작은 조각(chunk)으로 분할했습니다.")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        print(f"총 {len(split_documents)}개의 페이지(Document)를 임베딩합니다...")
        # 분할된 문서를 기반으로 벡터 DB를 생성합니다.
        vector_db = FAISS.from_documents(split_documents, embedding_model)
        
        vector_db.save_local(FAISS_DB_PATH)
        print(f"✅ 새로운 벡터 DB 구축 및 저장 완료: '{FAISS_DB_PATH}'")

    # 검색기는 그대로 사용
    return vector_db.as_retriever(search_kwargs={'k': 3})


# --- 3. Utility and Prompt Functions ---

def is_multiple_choice(question_text: str) -> bool:
    """Determines if a question is multiple-choice or open-ended."""
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?[\.\s]", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> tuple[str, list[str]]:
    """Separates the core question and the list of choices in a multiple-choice question."""
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?[\.\s]", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options

def make_prompt(text: str) -> str:
    """Creates a prompt for generating an answer using only the model's internal knowledge."""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = f"""### Instruction:
Provide only the number of the correct answer for the following question. Do not add any other explanations.

### Question:
{question}

### Choices:
{chr(10).join(options)}

### Answer:
"""
    else:
        prompt = f"""### Instruction:
Describe the answer to the following question in a complete Korean sentence, focusing on core keywords. Use your background knowledge even if not explicitly mentioned in any reference document. Do not use phrases like "According to the document...".

### Question:
{text}

### Answer:
"""
    return prompt

def make_rag_prompt(text: str, context: str) -> str:
    """Creates a prompt that includes the retrieved RAG context."""
    if not context.strip():
        return make_prompt(text)

    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = f"""### Instruction:
Based on the 'Reference Documents' provided, output only the number of the correct answer for the following question. Do not add any other explanations.

### Reference Documents:
{context}

### Question:
{question}

### Choices:
{chr(10).join(options)}

### Answer:
"""
    else:
        prompt = f"""### Instruction:
Based on the content of the 'Reference Documents', describe the answer to the following question in a complete Korean sentence. Do not use expressions like "According to the Reference Documents". Use your background knowledge even if not explicitly mentioned.

### Reference Documents:
{context}

### Question:
{text}

### Answer:
"""
    return prompt

# --- 4. Model and Tokenizer Loading ---

print("⏳ Loading the model and tokenizer...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ★★★ Directly load the SOLAR model ★★★
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- LoRA loading and merging section is removed ---
# print(f"⏳ Loading LoRA adapter from '{LORA_ADAPTER_PATH}' and applying to the model...")
# try:
#     model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
# except Exception as e:
#     print(f"❌ Error: Failed to load LoRA adapter. Please check the path: {LORA_ADAPTER_PATH}")
#     print(e)
#     exit()
# print("⏳ Merging LoRA weights into the base model...")
# model = model.merge_and_unload()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("✅ Model loading and setup complete.")


# --- 5. Post-processing and Helper Functions ---

def post_process_answer(generated_text: str, original_question: str) -> str:
    """Enhanced function to extract and clean the final answer from the generated text."""
    answer = generated_text.strip()
    
    if not answer:
        return "1"  # Default to '1' if the answer is empty

    if "###" in answer:
        answer = answer.split("###")[0].strip()
        
    if is_multiple_choice(original_question):
        # Step 1: Extract number from clear patterns like "The answer is 5"
        match = re.search(r'(?:정답은|답은|선택은)\s*\D*(\d+)', answer)
        if match:
            return match.group(1)

        # Step 2: Extract number from patterns like "5번", "5."
        match = re.search(r'\b(\d+)\s*(?:번|번입니다|\.)', answer)
        if match:
            return match.group(1)

        # Step 3: Extract number at the beginning of the string
        match = re.search(r"^\s*(\d+)", answer)
        if match:
            return match.group(1)

        # Step 4: If no match yet, find the first number in the entire text
        match = re.search(r'(\d+)', answer)
        if match:
            return match.group(1)
            
        # Step 5: If still no number is found, default to '1'
        return "1"
    
    return answer if answer else "Failed to generate an answer."

def is_code_detected(text: str) -> bool:
    """Checks for code snippets in the generated text based on simple keywords."""
    code_keywords = ['def ', 'import ', 'class ', 'r\'', 'sys.stdout', 'ans_qna']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in code_keywords)


# --- 6. Main Execution with RAG ---
if __name__ == "__main__":
    retriever = build_or_load_rag_backend()

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"✅ Successfully loaded test data from '{TEST_CSV_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{TEST_CSV_PATH}'. Please check the file path.")
        exit()

    preds = []
    MAX_RETRIES = 3 

    for index, q in tqdm(enumerate(test_df['Question']), total=len(test_df), desc="🚀 RAG inference in progress"):
        
        try:
            retrieved_docs = retriever.get_relevant_documents(q)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"⚠️ An error occurred during document retrieval for TEST_{index}: {e}")
            context_text = ""

        prompt = make_rag_prompt(q, context_text)
        
        is_valid_answer = False
        retries = 0
        generated_text = ""

        while not is_valid_answer and retries < MAX_RETRIES:
            if retries > 0:
                print(f"\n🔄 Retrying answer generation for TEST_{index}... ({retries}/{MAX_RETRIES})")

            output = pipe(
                prompt, 
                max_new_tokens=512,
                temperature=0.1 + (retries * 0.15),
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = output[0]['generated_text']

            if is_code_detected(generated_text):
                retries += 1
                if retries == MAX_RETRIES:
                    print(f"❌ Exceeded max retries for TEST_{index}. Using the last generated answer.")
                    is_valid_answer = True
            else:
                is_valid_answer = True

        pred_answer = post_process_answer(generated_text, original_question=q)
        preds.append(pred_answer)

    print("\n📄 Inference complete. Generating submission file...")
    try:
        sample_submission = pd.read_csv('/workspace/open/sample_submission.csv')
        sample_submission['Answer'] = preds
        sample_submission.to_csv(SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ Submission file created successfully: '{SUBMISSION_CSV_PATH}'")
    except FileNotFoundError:
        print(f"❌ Error: '/workspace/open/sample_submission.csv' not found. Please check the file path.")