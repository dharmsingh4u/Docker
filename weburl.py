from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langgraph.graph import StateGraph,START, END,MessagesState
from typing import TypedDict,Annotated,Literal
import os 
import sys
#sys.path.insert(1, r'D:\Notebooks\LLM\env')
#from enviorment import load_env
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
#from pydirectoryloader import rag_function
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from pydantic import BaseModel,Field
def get_ruls(start_url,max_urls):

# A set to store unique URLs we have visited
    visited_urls = set()
    # A queue (list) to store URLs we still need to visit
    urls_to_visit = []
    # The set to store the final list of internal links
    internal_urls = set()

    urls_to_visit.append(start_url)
    domain_name = urlparse(start_url).netloc

    while urls_to_visit and len(internal_urls) < max_urls:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        print(f"Fetching: {current_url}")
        visited_urls.add(current_url)

        try:
            response = requests.get(current_url, timeout=5)
            response.raise_for_status() # Check for bad status codes
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {current_url}: {e}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        for anchor_tag in soup.find_all('a', href=True):
            href = anchor_tag['href']
            # Join relative URLs with the base URL to make them absolute
            full_url = urljoin(current_url, href)
            # Parse the URL to verify its domain
            parsed_url = urlparse(full_url)

            # Ensure the link is internal and valid
            if parsed_url.netloc == domain_name and parsed_url.scheme in ['http', 'https']:
                # Clean the URL to ignore fragments (e.g., #section)
                clean_url = urljoin(full_url, urlparse(full_url).path)
                
                if clean_url not in internal_urls:
                    internal_urls.add(clean_url)
                    urls_to_visit.append(clean_url)
                    #print(f"  Found new URL: {clean_url}")
        
        # Be respectful: add a small delay between requests
        #time.sleep(1)
        return list(internal_urls)
class output(BaseModel):
    urls :list[str]=Field(description='contains the list of the relevant url')
def URL_selector (model,question,website):
    model_stuctured=model.with_structured_output(output)

    prompt =ChatPromptTemplate.from_messages([('system','you are URL selector who helps getting the relevant url from the list for a given question '),
                                          ('human','for a given question {question} , get me the relevant top 10 urls  from the URLS list {URL} which can help to answer the question ')])
    chain=prompt| model_stuctured
    l=get_ruls(website,100)
    result=chain.invoke({'question':question,"URL":l})
    return result.urls
def URL_loader(urls):
    #loader = WebBaseLoader(urls)
    #docs_lazy = loader.lazy_load()
    #docs=[]
    #for doc in docs_lazy:
    #    docs.append(doc)
    all_docs=[]
    for url in urls:
        loader = FireCrawlLoader(
            api_key="fc-418ca318825e406fb656fd3c591812d8",
            url=url,
            mode="scrape"  # only scrape single page
        )
        docs = loader.load()
        all_docs.extend(docs)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    #documents=text_splitter.split_documents(docs)
    documents=text_splitter.split_documents(all_docs)
    db=FAISS.from_documents(documents, OpenAIEmbeddings())
    #db.save_local('vectorstore_index')
    retriever=db.as_retriever(search_type='similarity',search_kwargs={"k":3})
    return retriever
def retriver_questions(result):
    #result=retriever.invoke(question)
    context_text= "\n".join([doc.page_content for doc in result])
    return context_text

