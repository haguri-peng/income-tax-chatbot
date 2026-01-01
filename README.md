# 🤖 소득세 챗봇

제이쓴 님의 [RAG를 활용한 LLM Application 개발 (feat. LangChain)](https://www.inflearn.com/course/rag-llm-application%EA%B0%9C%EB%B0%9C-langchain/dashboard) 강의를 참조하여 작업한 내용이며, 수업에서 다루는 대한민국 `소득세법`에 대한 질문에 정확하고 전문적인 답변을 제공하는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.

LangChain을 활용해 Pinecone 벡터 DB에서 소득세법 문서를 검색하고, xAI의 Grok 모델(grok-4-1-fast-reasoning)을 사용해 자연스럽고 법령에 근거한 답변을 생성합니다.

![챗봇 화면 예시](./images/income-tax-chatbot-example.png)

## 🧰 주요 기능

- 소득세법 관련 질문에 대해 법조항(조문)을 인용하며 간결하게 답변
- 질문 재작성 기능 ("직장인" → "거주자" 등 전문 용어로 변환)
- Few-shot 프롬프트로 답변 스타일 통일 (예: "소득세법 제XX조에 따르면...")
- 실시간 스트리밍 답변 (Streamlit `st.write_stream` 활용)
- 대화 히스토리 유지 (세션 기반 메모리)

## 📚 기술 스택

- **프론트엔드**: Streamlit
- **LLM**: xAI Grok (`grok-4-1-fast-reasoning`)
- **임베딩 모델**: dragonkue/multilingual-e5-small-ko (HuggingFace)
- **벡터 DB**: Pinecone (`tax-markdown-index-small`)
- **프레임워크**: LangChain
- **기타**: python-dotenv, langchain-pinecone, langchain-huggingface, langchain-xai

## 🧭 설치 및 실행 방법

### 1. 리포지토리 클론
```bash
git clone https://github.com/your-username/income-tax-chatbot.git
cd income-tax-chatbot
```

### 2. 가상환경 생성 및 패키지 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 예시 (필수 패키지):
```
streamlit
langchain
langchain-xai
langchain-pinecone
langchain-huggingface
langchain-community
pinecone-client
python-dotenv
```

### 3. 환경 변수 설정
`.env` 파일을 프로젝트 루트에 생성하고 아래 내용을 추가하세요.

```
PINECONE_API_KEY=your-pinecone-api-key
XAI_API_KEY=your-xai-api-key  # 필요 시
```

### 4. Pinecone 인덱스 준비
- 인덱스 이름: `tax-markdown-index-small`
- `tax_with_markdown.docx` 문서(소득세 정보가 있는 파일)를 기반으로 이미 업로드된 인덱스를 사용합니다.
- 새로 인덱싱이 필요하다면 별도로 작업한 노트북을 참고해 문서를 인덱싱하여 업로드해야 합니다.

### 5. 앱 실행
```bash
streamlit run chat.py
```

브라우저에서 http://localhost:8501 로 접속하면 챗봇을 바로 사용할 수 있습니다.

## 🎄 프로젝트 구조

```
.
├── chat.py                  # Streamlit 메인 앱
├── llm.py                   # LangChain RAG 체인 및 LLM 설정
├── config.py                # Few-shot 답변 예시
├── .env                     # 환경 변수 (Git에 올리지 마세요)
└── requirements.txt
```

## 💡 사용 예시

**질문**: 연봉이 5천만원인 직장인의 종합 소득세를 알려줄래요? 산출과정까지 보여주면 좋겠습니다.
**답변 예시**:
```
소득세법 제47조에 따라 총급여 5천만원에서 근로소득공제 1,225만원을 뺀 근로소득금액 3,775만원, 기본공제 150만원(본인 가정)을 적용해 과세표준 3,625만원이 됩니다. 제55조 세율에 따라 84만원 + (2,225만원×15%)=417만7,500원이 산출세액입니다. 실제 납부세액은 세액공제 등으로 줄어듭니다.
```

---