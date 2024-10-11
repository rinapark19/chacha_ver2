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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunking(pdf_list, n, chunk_length=2048):
    chunks = []
    with open("data/rag_data/spiderman.txt", "r", encoding="utf-8") as f:
        text = f.read()
        chunks.extend([text[i:i+chunk_length] for i in range(0, len(text), chunk_length)])
        
    return chunks

def get_retriever(chunks, embedding_model="text-embedding-ada-002", retrieve_function="similarity"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    embedding_vectors = [embeddings.embed_documents]