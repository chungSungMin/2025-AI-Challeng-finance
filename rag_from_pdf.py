import os
import torch
import transformers
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# langchain_community.embeddings -> langchain_huggingface로 변경
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline # 실제 LLM 연동을 위해 추가
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ----------------------------------------------------------------
# ✨ 1. 실제 LLM을 로드하는 함수 (최초 코드 활용)
# ----------------------------------------------------------------
def setup_model():
    """
    HuggingFace에서 4비트 양자화된 실제 LLM과 토크나이저를 로드합니다.
    """
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

# ----------------------------------------------------------------
# ✨ 2. 로드된 모델을 LangChain과 연동하는 부분
# ----------------------------------------------------------------
def load_real_llm():
    model, tokenizer = setup_model()
    if model is None or tokenizer is None:
        return None

    # Transformers 라이브러리의 파이프라인 생성
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024 # 답변 생성 최대 길이
    )

    # LangChain에서 사용할 수 있도록 파이프라인을 래핑
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# --- RAG 파이프라인 시작 ---

# --- 단계 1: PDF 로드 ---
print("--- [단계 1: PDF 로드] ---")
pdf_folder_path = "./data"
documents = []
if not os.path.exists(pdf_folder_path):
    print(f"경고: '{pdf_folder_path}' 폴더를 찾을 수 없습니다. PDF를 로드할 수 없습니다.")
else:
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
print(f"✅ 총 {len(documents)}개의 페이지를 로드했습니다.")
# ✨ [확인 코드] 로딩된 첫 페이지 내용 확인
if documents:
    print(f"📄 첫 페이지 출처: {documents[0].metadata.get('source', 'N/A')}")
    print(f"📄 첫 페이지 내용 일부: {documents[0].page_content[:200]}...\n")
print("-" * 50)


# --- 단계 2: 청크 분할 ---
print("--- [단계 2: 청크 분할] ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(documents)
print(f"✅ 문서를 총 {len(splits)}개의 청크로 분할했습니다.")
# ✨ [확인 코드] 분할된 첫 청크 내용 확인
if splits:
    print(f"📄 첫 번째 청크 내용: {splits[0].page_content[:300]}...\n")
print("-" * 50)


# --- 단계 3: 임베딩 및 벡터 DB 저장 ---
print("--- [단계 3: 임베딩 및 벡터 DB 저장] ---")
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)
print("✅ 벡터 DB 저장 완료.\n")
print("-" * 50)


# --- 단계 4 & 5: RAG 체인 구성 및 실행 ---
print("--- [단계 4/5: RAG 체인 구성 및 실행] ---")
retriever = vectordb.as_retriever()

# ✨ [확인 코드] 리트리버 단독 테스트
print("🕵️ 리트리버가 질문과 관련된 문서를 잘 찾아오는지 테스트합니다...")
test_question = "북한 해킹 그룹의 주요 공격 방식은 무엇인가요?"
try:
    relevant_docs = retriever.invoke(test_question)
    print(f"\n[테스트 질문]: '{test_question}'")
    for i, doc in enumerate(relevant_docs):
        print(f"\n--- 관련 문서 #{i+1} ---")
        print(f"출처: {doc.metadata.get('source', 'N/A')}")
        print(f"내용: {doc.page_content[:200]}...")
except Exception as e:
    print(f"리트리버 테스트 중 오류 발생: {e}")
print("\n" + "-" * 50)


print("⚙️ 실제 LLM을 로드하여 RAG 체인을 최종 구성합니다...")
# 실제 LLM 로드
llm = load_real_llm()

if llm:
    template = """
    당신은 대한민국 최고의 사이버 보안 전문가입니다. 주어진 [Context] 정보만을 사용하여 다음 [Question]에 대해 명확하고 전문가적으로 답변하십시오.
    답변은 반드시 한국어로 작성해야 합니다. 만약 [Context]에 답변에 필요한 정보가 없다면, "제공된 정보만으로는 답변할 수 없습니다."라고만 답변하십시오.

    [Context]
    {context}

    [Question]
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- [최종 실행] ---")
    question_to_ask = "북한 해킹 그룹 라자루스의 주요 활동 방식에 대해 설명해줘."
    print(f"질문: {question_to_ask}")
    
    # RAG 체인 실행
    response = rag_chain.invoke(question_to_ask)

    print("\n--- [최종 결과] ---")
    print(f"답변: {response}")
    print("-" * 50)
else:
    print("LLM 로드에 실패하여 RAG 체인을 실행할 수 없습니다.")