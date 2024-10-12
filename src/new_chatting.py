from pypdf import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import time
import faiss

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunking(chunk_length=100):
    chunks = []
    with open("data/rag_data/spiderman.txt", "r", encoding="utf-8") as f:
        text = f.read()
        chunks.extend([text[i:i+chunk_length] for i in range(0, len(text), chunk_length)])
        
    return chunks

def get_retriever(chunks, embedding_model="text-embedding-ada-002", retrieve_function="similarity"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    text_embeddings = embeddings.embed_documents(chunks)
    text_embeddings = np.array(text_embeddings).astype('float32')

    d = text_embeddings.shape[1]
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    index.train(text_embeddings)
    index.add(text_embeddings)

    docstore = InMemoryDocstore({i: Document(page_content=text) for i, text in enumerate(chunks)})
    index_to_docstore_id = {i: i for i in range(len(chunks))}
    vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

    return vectorstore.as_retriever()

def get_chain(retriever, model_name="gpt-4-1106-preview"):
    prompt = hub.pull("rlm/rag-prompt")
    rag_llm = ChatOpenAI(
        model_name=model_name,
        temperature=0
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=rag_llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        verbose = True,
        return_source_documents=True
    )
    
    return qa_chain

if __name__ == "__main__":
    chunks = chunking()
    retriever = get_retriever(chunks)
    chain = get_chain(retriever)

    start_time = time.time()
    result = chain({"query": "얼티밋 유니버스가 뭐야?"})
    end_time = time.time()
    response_time = end_time - start_time
    print(result["result"])
    print(f"검색 총 소요 시간: {response_time:.2f}")