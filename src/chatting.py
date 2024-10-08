from pypdf import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import time

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from util import MODEL_LIST, PROMPT_LIST

def chunking(pdf_list, n, chunk_length=2048):
    chunks = []
    for i in range(n):
        reader = PdfReader(pdf_list[i])
        
        for page in reader.pages:
            text_page = page.extract_text()
            chunks.extend([text_page[i: i+chunk_length] for i in range(0, len(text_page), chunk_length)])
            
    return chunks

def get_retriever(chunks, embedding_model="text-embedding-ada-002", retrieve_function="similarity"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever

def get_tool(retriever, model_name="gpt-4-1106-preview"):
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
    llm = ChatOpenAI(
        model_name = model_name,
        temperature=1
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
        agent_kwargs={"system_message": system_message},
        handle_parsing_errors=True
    )
    
    return agent
    
class persona_agent:
    def __init__(self, pdf_list, char) -> None:
        self.chunks = chunking(pdf_list, len(pdf_list))
        self.retrieve = get_retriever(self.chunks)
        self.tools = get_tool(self.retrieve, MODEL_LIST[char])
        system_message = PROMPT_LIST[char]
        self.agent = get_agent(system_message, self.tools, MODEL_LIST[char])
    
    def receive_chat(self, chat):
        while True:
            start_time = time.time()
            result = self.agent.run(chat)
            end_time = time.time()
            
            response_time = end_time - start_time
            return result
        
if __name__ == "__main__":
    pdf_list = ["data/rag_data/jwc.pdf"]
    char = "jwc"
    agent = persona_agent(pdf_list, char)
    
    print(agent.receive_chat("넌 누구야?"))