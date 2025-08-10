import os
import torch
import json
import re
import random # 주제 다양성을 위해 추가
import time   # 생성 시간 측정을 위해 추가
import transformers
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Pydantic 모델은 변경 없이 그대로 사용합니다.
class MultipleChoiceQuestion(BaseModel):
    question: str = Field(description="생성된 객관식 질문")
    options: dict[str, str] = Field(description="키가 '1', '2', '3', '4'인 4개의 선택지 딕셔너리")
    answer: str = Field(description="정답에 해당하는 선택지의 키 ('1'~'4')")

# setup_model, load_real_llm 함수는 변경 없이 그대로 사용합니다.
def setup_model():
    """HuggingFace에서 4비트 양자화된 LLM과 토크나이저를 로드합니다."""
    print("🚀 LLM 모델 로드를 시작합니다... (시간이 걸릴 수 있습니다)")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "K-intelligence/Midm-2.0-Base-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config
        )
        print("✅ LLM 및 토크나이저 로드 완료.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return None, None

def load_real_llm(model, tokenizer):
    """로드된 모델을 LangChain과 연동 가능한 파이프라인으로 만듭니다."""
    if model is None or tokenizer is None:
        return None
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        repetition_penalty=1.2,
        temperature=0.8,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# ✨ [핵심 수정 1] rerank_documents 함수가 미리 생성된 cross_encoder 모델을 인자로 받도록 변경
def rerank_documents(cross_encoder: CrossEncoder, question: str, documents: list, top_n=3):
    """CrossEncoder를 사용하여 관련성이 높은 문서를 재정렬합니다."""
    pairs = [[question, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in scored_docs[:top_n]]


def main():
    """개선된 RAG 파이프라인으로 300개의 퀴즈를 생성하고 JSON 파일로 저장합니다."""
    
    NUM_QUIZZES_TO_GENERATE = 300
    OUTPUT_FILENAME = "신용정보의이용및보호_300.json"

    print("--- [준비 단계: RAG 시스템 초기화] ---")
    
    pdf_folder_path = "./data"
    if not os.path.exists(pdf_folder_path) or not any(f.endswith('.pdf') for f in os.listdir(pdf_folder_path)):
        print(f"❌ 에러: '{pdf_folder_path}' 폴더가 없거나 폴더 안에 PDF 파일이 없습니다.")
        return
        
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            print(f"📄 '{file}' 파일 로드 중...")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        print("❌ 문서를 분할하지 못했습니다. PDF 파일 내용을 확인해주세요.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    
    print("✅ RAG 시스템 초기화 완료.\n" + "=" * 50)

    model, tokenizer = setup_model()
    llm = load_real_llm(model, tokenizer)
    if not llm:
        print("LLM 로드에 실패하여 프로그램을 종료합니다."); return

    # ✨ [핵심 수정 2] CrossEncoder 모델을 루프 시작 전에 딱 한 번만 로드합니다.
    print("🚀 Reranker 모델 로드를 시작합니다...")
    try:
        reranker = CrossEncoder('dragonkue/bge-reranker-v2-m3-ko', device='cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ Reranker 모델 로드 완료.")
    except Exception as e:
        print(f"❌ Reranker 모델 로드 중 오류 발생: {e}")
        return

    generated_quizzes = []
    
    start_time = time.time()
    for i in range(NUM_QUIZZES_TO_GENERATE):
        print(f"\n{'='*20} 퀴즈 {i+1}/{NUM_QUIZZES_TO_GENERATE} 생성 시작 {'='*20}")
        
        llm_output = ""
        try:
            seed_chunk = random.choice(splits)
            seed_query = seed_chunk.page_content[:200]
            
            retrieved_docs = retriever.invoke(seed_query)
            
            if len(retrieved_docs) < 3:
                context_docs = [seed_chunk]
            else:
                # ✨ [핵심 수정 3] 미리 로드된 reranker 모델을 인자로 전달합니다.
                context_docs = rerank_documents(reranker, seed_query, retrieved_docs, top_n=3)

            context_string = "\n---\n".join([doc.page_content for doc in context_docs])

            prompt = ChatPromptTemplate.from_template(
                """
                [지시]
                당신은 유능한 교육 콘텐츠 제작 전문가입니다. 주어진 [문서 내용]을 바탕으로, 내용의 핵심 개념을 묻는 객관식 문제 1개를 생성해야 합니다.

                [작업 절차]
                1. **생각의 과정**: 먼저 [문서 내용]을 분석하여 문제 생성에 대한 계획을 단계별로 서술합니다.
                2. **최종 JSON 출력**: '생각의 과정'이 끝나면, 그 내용을 바탕으로 최종 결과물을 아래 [JSON 출력 형식]에 맞춰 JSON으로만 출력합니다. JSON 객체는 반드시 ```json ... ``` 코드 블록으로 감싸야 합니다.

                [문서 내용]
                {context}

                [JSON 출력 형식]
                - 최상위 객체는 "question", "options", "answer" 세 개의 키를 가져야 합니다.
                - "question"의 값은 문자열입니다.
                - "options"의 값은 반드시 키가 "1", "2", "3", "4"인 4개의 선택지를 포함하는 딕셔너리(객체)여야 합니다.
                - "answer"의 값은 정답에 해당하는 선택지의 키(예: "1") 문자열입니다.

                ---
                
                [생각의 과정]
                1. 핵심 개념 선정: 
                2. 질문 구성: 
                3. 선택지(options) 구성: 
                   - 정답 (키 "1"): 
                   - 오답 (키 "2"): 
                   - 오답 (키 "3"): 
                   - 오답 (키 "4"): 
                4. 최종 JSON 조합 및 정답 키 확인: 

                [최종 JSON 출력]
                ```json
                """
            )
            
            chain = prompt | llm | StrOutputParser()
            llm_output = chain.invoke({"context": context_string})
            
            json_str = None
            
            json_match = re.search(r"```json\n(.*?)\n```", llm_output, re.DOTALL)
            if json_match:
                content_inside_block = json_match.group(1).strip()
                first_brace_index = content_inside_block.find('{')
                last_brace_index = content_inside_block.rfind('}')
                if first_brace_index != -1 and last_brace_index != -1:
                    json_str = content_inside_block[first_brace_index : last_brace_index + 1]

            if not json_str:
                first_brace_index = llm_output.find('{')
                last_brace_index = llm_output.rfind('}')
                if first_brace_index != -1 and last_brace_index != -1:
                    json_str = llm_output[first_brace_index : last_brace_index + 1]

            if not json_str:
                raise ValueError("LLM 출력에서 유효한 JSON 객체를 찾을 수 없습니다.")
            
            quiz_data = json.loads(json_str)
            validated_quiz = MultipleChoiceQuestion(**quiz_data)
            quiz_json = validated_quiz.model_dump()

            generated_quizzes.append(quiz_json)
            print(f"✅ 퀴즈 {i+1} 생성 완료. 현재까지 총 {len(generated_quizzes)}개 생성.")

        except (ValidationError, json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"❌ 퀴즈 {i+1} 생성 실패: {e}")
            print("\n--- [LLM 원본 출력 (오류 원인)] ---")
            print(llm_output)
            print("---------------------------------")
            continue
        finally:
            # ✨ [핵심 수정 4] 루프가 끝날 때마다 캐시를 비워 메모리 조각화를 방지합니다.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*20} 모든 작업 완료 {'='*20}")
    if generated_quizzes:
        print(f"총 요청: {NUM_QUIZZES_TO_GENERATE}개")
        print(f"성공: {len(generated_quizzes)}개")
        print(f"실패: {NUM_QUIZZES_TO_GENERATE - len(generated_quizzes)}개")
        print(f"총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        
        print(f"\n생성된 퀴즈를 '{OUTPUT_FILENAME}' 파일로 저장합니다.")
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(generated_quizzes, f, ensure_ascii=False, indent=4)
        print("✅ 파일 저장 완료.")
    else:
        print("\n❌ 생성된 퀴즈가 없어 파일을 저장하지 않습니다.")

if __name__ == "__main__":
    main()
