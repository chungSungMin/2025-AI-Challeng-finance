import os
import torch
import json
import transformers
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# setup_model() 함수는 변경 없이 그대로 사용합니다.
def setup_model():
    """HuggingFace에서 4비트 양자화된 실제 LLM과 토크나이저를 로드합니다."""
    print("🚀 실제 LLM 모델 로드를 시작합니다... (시간이 걸릴 수 있습니다)")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
        )
        print("✅ 실제 LLM 및 토크나이저 로드 완료.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return None, None

# load_real_llm() 함수도 변경 없이 그대로 사용합니다.
def load_real_llm():
    """로드된 모델을 LangChain과 연동 가능한 파이프라인으로 만듭니다."""
    model, tokenizer = setup_model()
    if model is None or tokenizer is None:
        return None
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, repetition_penalty=1.2)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# ✨ 사용자가 제공한 문제 생성용 프롬프트 템플릿
def get_quiz_generation_prompt():
    return """
    당신은 정보보안기사 국가공인시험 출제위원입니다. 오직 주어진 [Context] 정보만을 사용하여, 수험생의 허를 찌르는 변별력 높은 문제를 만들어주세요.

    [기본 규칙]
    1. [Context]에 명시적으로 언급된 내용만을 근거로 질문과 선지, 해설을 생성해야 합니다.
    2. 4개의 선지를 생성하며, 그중 정답은 반드시 하나여야 합니다.
    3. 추측에 기반하거나 [Context]에 없는 불확실한 정보로 선지나 해설을 만들어서는 안 됩니다.

    [금지 사항]
    - '다음 중 성격이 다른 하나는?' 과 같이 기준이 모호한 질문은 생성하지 마세요.
    - 단순히 [Context]의 문장을 복사-붙여넣기 한 것처럼 보이는 선지는 만들지 마세요.

    [해설 작성 규칙]
    - 'explanation' 항목에는 정답의 근거를 [Context] 내용에 기반하여 명확히 제시해야 합니다.
    - 정답 해설뿐만 아니라, 나머지 오답 선지들이 왜 틀렸는지에 대한 간략한 설명도 포함해야 합니다.

    [세분화된 JSON 형식]
    - 반드시 아래 [JSON 형식]을 엄격하게 준수하여 답변해야 합니다. 코드 블록(` ```json ... ``` `)으로 감싸서 출력해주세요.
    {{
        "topic": "{topic}",
        "question": "생성된 질문 내용",
        "options": {{
            "1": "첫 번째 선지",
            "2": "두 번째 선지",
            "3": "세 번째 선지",
            "4.": "네 번째 선지"
        }},
        "answer": "정답 선지의 번호 (예: '1')",
        "explanation": {{
            "correct_reason": "정답이 맞는 이유에 대한 상세한 해설",
            "incorrect_reasons": {{
                "1": "1번 선지가 오답인 이유",
                "2": "2번 선지가 오답인 이유",
                "3": "3번 선지가 오답인 이유"
            }}
        }}
    }}
    ---
    [실제 생성 요청]
    
    [Topic]: {topic}

    [Context]:
    {context}
    """

def main():
    """RAG 파이프라인을 설정하고 지정된 주제에 대해 문제를 생성하는 메인 함수"""

    # PDF 로드, 청크 분할, DB 저장 과정은 동일합니다.
    print("--- [단계 1: PDF 로드, 분할, DB 저장] ---")
    pdf_folder_path = "./data"
    documents = []
    if not os.path.exists(pdf_folder_path):
        print(f"경고: '{pdf_folder_path}' 폴더를 찾을 수 없습니다.")
        return
        
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectordb.as_retriever()
    print("✅ RAG 준비 완료.\n")
    print("-" * 50)

    # --- 단계 2: 문제 생성 실행 ---
    print("--- [단계 2: 문제 생성 실행] ---")
    
    # ✨✨✨ 여기에 문제를 생성하고 싶은 '주제'를 지정합니다. ✨✨✨
    topic_to_generate = "북한의 사이버 공격 전술"
    print(f"주제: '{topic_to_generate}'에 대한 문제 생성을 시작합니다.")
    
    # 1. 주제 기반으로 관련성 높은 문서(Context) 검색
    retrieved_docs = retriever.invoke(topic_to_generate)
    # 검색된 문서들의 내용을 하나의 문자열로 합칩니다.
    context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    print(f"\n[검색된 Context 일부]:\n{context_string[:500]}...\n")

    # 실제 LLM 로드
    llm = load_real_llm()
    if not llm:
        print("LLM 로드에 실패하여 프로그램을 종료합니다.")
        return

    # 2. 검색된 Context를 프롬프트에 주입하여 문제 생성 요청
    prompt_template = get_quiz_generation_prompt()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 체인 구성: 이제 체인은 retriever를 직접 사용하지 않습니다.
    quiz_generation_chain = prompt | llm | StrOutputParser()
    
    # 체인 실행
    response = quiz_generation_chain.invoke({
        "topic": topic_to_generate,
        "context": context_string
    })

    print("\n--- [최종 생성 결과] ---")
    print("LLM이 생성한 원본 출력:")
    print(response)
    
    # ✨ [추가] 생성된 JSON 문자열을 파싱하여 깔끔하게 출력
    try:
        # 모델 출력에서 JSON 코드 블록만 추출
        json_str = response.split("```json")[1].split("```")[0].strip()
        quiz_data = json.loads(json_str)
        print("\n✅ JSON 파싱 성공! 깔끔하게 정리된 결과:")
        print(json.dumps(quiz_data, indent=2, ensure_ascii=False))
    except (IndexError, json.JSONDecodeError) as e:
        print(f"\n❌ 오류: 모델이 생성한 결과에서 유효한 JSON을 파싱하는 데 실패했습니다. (오류: {e})")

    print("-" * 50)


if __name__ == "__main__":
    main()