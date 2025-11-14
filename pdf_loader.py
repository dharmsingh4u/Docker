import sys
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
#sys.path.insert(1, r'D:\Notebooks\LLM\env')
#from enviorment import load_env
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
load_env()
model =ChatOpenAI()
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma,FAISS
def loader(path):
    loader=PyPDFLoader(path)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents=text_splitter.split_documents(docs)
    
    db = FAISS.from_documents(documents,OpenAIEmbeddings())
    retriever=db.as_retriever(search_type='similarity',search_kwargs={"k":4})
    return retriever
