from openai import OpenAI
import os
from dotenv import load_dotenv
import time

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever

import openai
import json
import re 

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from util import MODEL_LIST, PROMPT_LIST

def chunking(data):
    ''' Recursive Character Splitter을 활용해 주어진 데이터를 청크로 분할 '''
    with open(data, "r", encoding="utf-8") as f:
        file = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    return text_splitter.split_text(file)

def get_cached_embedder():
    ''' Cached Backed Embeddings로 임베더 정의 '''
    underlying_embeddings = OpenAIEmbeddings()
    store = LocalFileStore("data/cache/")
    return CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store, namespace=underlying_embeddings.model)

def get_vectorstore(chunks, embedder):
    ''' 분할한 청크, 임베더로 벡터스토어 정의 '''
    return FAISS.from_texts(chunks, embedder)

def get_retriever(embedder, vectorstore):
    ''' retriever 파이프라인 구성 후 retriever 정의 '''
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embedder)
    relevant_filter = EmbeddingsFilter(embeddings=embedder, similarity_threshold=0.76)

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    retriever = vectorstore.as_retriever()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

    return compression_retriever


def get_tool(retriever, model_name="gpt-4-1106-preview"):
    ''' retriever로 QA chain 및 agent의 tool 정의 '''
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """)
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
    
    tools = [
        Tool(
            name="doc_search_tool",
            func=qa_chain,
            description=(
                "This tool is used to retrieve information from the knowledge base"
            )
        )
    ]
    
    return tools

def get_agent(system_message, tools, model_name):
    ''' agent 정의 '''
    llm = ChatOpenAI(
        model_name = model_name,
        temperature=1,
        streaming=True # agent 성능 개선
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=True
    )
    
    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=memory,
        return_source_documents=True,
        return_intermediated_steps=True,
        agent_kwargs={"system_message": system_message,
                      "max_iteration": 3, # agent 성능 개선
                      "max_execution_time": 10.0,
                      "trip_intermediate_steps": 1},
        handle_parsing_errors=True
    )
    
    return agent

def moderate_content(text):
    ''' agent의 출력에서 폭력적/혐오적 표현 감지 '''
    response = openai.moderations.create(input=text, model="omni-moderation-latest")
    
    if response.results[0].flagged:
        return "부적절한 콘텐츠가 감지되었습니다. 문장을 다시 입력해 주세요."
    
    with open("data/badwords.json", "r", encoding="utf-8") as f:
        json_data = f.read()
    violent_words = json.loads(json_data)
    
    for word in violent_words:
        if re.search(r'\b{}\b'.format(re.escape(word)), text, re.IGNORECASE):
            return "부적절한 콘텐츠가 감지되었습니다. 문장을 다시 입력해 주세요."
    
    return text
    
class persona_agent:
    def __init__(self, pdf_list, char) -> None:
        self.chunks = chunking(pdf_list)
        self.embedder = get_cached_embedder()
        self.vectorstore = get_vectorstore(self.chunks, self.embedder)
        self.retriever = get_retriever(self.embedder, self.vectorstore)
        self.tools = get_tool(self.retriever, MODEL_LIST[char])
        system_message = PROMPT_LIST[char]
        self.agent = get_agent(system_message, self.tools, MODEL_LIST[char])
    
    def receive_chat(self, chat):
        while True:
            start_time = time.time()
            result = self.agent.run(chat)
            end_time = time.time()
            
            response_time = end_time - start_time
            print(response_time)
            return result