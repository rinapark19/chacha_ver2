from pypdf import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import time

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.firestore import FirestoreChatMessageHistory

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=rag_llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
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

def get_agent(system_message, tools, model_name="gpt-4-1106-preview"):
    llm = ChatOpenAI(
        model_name = "gpt-4-1106-preview",
        temperature=1
    )
    
    firestore_config = {
            "collection_name": "chathistory",
            "session_id": 1,
            "user_id": 1,
    }
    
    chat_history = FirestoreChatMessageHistory(**firestore_config)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        return_message=True,
        output_key="output",
        chat_memory=chat_history
    )
    
    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=memory,
        return_source_documents=True,
        return_intermediated_steps=True,
        agent_kwargs={"system_message": system_message}
    )
    
    return agent
    
class persona_agent:
    def __init__(self, pdf_list) -> None:
        self.chunks = chunking(pdf_list, len(pdf_list))
        self.retrieve = get_retriever(self.chunks)
        self.tools = get_tool(self.retrieve)
        
        system_message = """
            You will act as a Spider-man(Peter Parker) character and answer the user's questions

            You can refer to the following information about Spider-man
            - Characteristic: The protagonist of The Amazing Spider-man, real name is Peter Parker.
            - Personality: Cheerful, cheeky, witty, brave, kind, and friendly.
            - Line: '무슨 신, 반짝이 신?', '아니, 널 지켜 주려고 그랬던 거야.', '걱정하지 마, 괜찮을 거야.', '비상사다리 타고. 별거 아니던걸, 뭐.' 

            You should follow the guidelines below:
            - If the answer isn't available within in the context, state the fact.
            - Otherwise, answer to your best capability, referring to source of documents provided.
            - Limit responses to three or four sentences for clarity and conciseness.
            - You must answer in Korean.
            - You must answer like you're Spider-man. Use a first-person perspective. Do not say "Peter Parker ~"
            - You must follow the Spider-man style naturally.
            - You must refer to source of documents provided to answer about Peter Parker
            - You must not change the answer from the documents provided.
        """
        
        self.agent = get_agent(system_message, self.tools)
    
    def receive_chat(self, chat):
        while True:
            start_time = time.time()
            result = self.agent(chat)
            end_time = time.time()
            
            response_time = end_time - start_time
            answer = result["output"]
            return answer
        
if __name__ == "__main__":
    pdf_list = ["data/spiderman1.pdf", "data/spiderman2.pdf"]
    agent = persona_agent(pdf_list)
    
    agent.receive_chat("안녕")