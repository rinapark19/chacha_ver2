## Langchain을 활용한 가상의 인격을 가진 캐릭터 챗봇 서비스 개발

**서비스 소개**
- RAG를 연결해 hallucination을 감소하고, 파인 튜닝과 프롬프트 엔지니어링을 활용해 캐릭터의 인격을 학습한 페르소나 챗봇 서비스입니다.

**배포 링크**: https://lcchatbotdist-gniwdbu9jtvealn58eyl35.streamlit.app/

**Skills & Frameworks**
- Python
- Langchain, Streamlit
- GPT-4

**File 구조**
```
chacha
├─ .gitignore
├─ data (모델 파인 튜닝 및 벡터스토어 데이터)
│  ├─ badwords.json
│  ├─ ft_data
│  │  ├─ data.ipynb
│  │  ├─ juc_line.csv
│  │  ├─ juc_lines.jsonl
│  │  ├─ line.tsv
│  │  ├─ pp_lines.jsonl
│  │  ├─ spiderman.jsonl
│  │  ├─ szg_line.csv
│  │  └─ szg_lines.jsonl
│  └─ rag_data
│     ├─ jwc.pdf
│     ├─ jwc.txt
│     ├─ spiderman.txt
│     ├─ spiderman1.pdf
│     ├─ spiderman2.pdf
│     ├─ szg.txt
│     ├─ szg1.pdf
│     ├─ szg2.pdf
│     └─ szg3.pdf
├─ experiments.ipynb (최적화 실험)
├─ practice2.ipynb
├─ README.md
└─ src
   ├─ chatting.py (Agent 정의 및 대화)
   ├─ data (배포용 데이터)
   │  ├─ badwords.json
   │  ├─ jwc.txt
   │  ├─ spiderman.txt
   │  └─ szg.txt
   ├─ debugging.py
   ├─ main.py (Streamlit 구현)
   ├─ prompt.txt
   └─ util.py (Agent에 필요한 util)

```