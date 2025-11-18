import streamlit as st
import requests
import tempfile
import os
import dotenv
dotenv.load_dotenv()
# You may need a library like 'pdfplumber' or 'PyPDF2' to process the PDF content
# pip install pdfplumber 
import pdfplumber 
import sys
#from dotenv import load_dotenv
#sys.path.insert(1, r'D:\Notebooks\Langraph\Rag\URL Rag')
#ys.path.insert(1, r'D:\Notebooks\Langraph\Rag\PDF-Rag-Agentic')
from pdf_loader import loader
from weburl import URL_selector,URL_loader,retriver_questions
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langgraph.graph import StateGraph,START, END,MessagesState
from typing import TypedDict,Annotated,Literal
import os 
from langchain_core.prompts import HumanMessagePromptTemplate
import sys
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
st.title("URL or PDF Upload App")
def clear_history_callback():
    """Deletes the chat history from session state."""
    if 'message_history' in st.session_state:
        del st.session_state['message_history']
# 1. Use a radio button to select the input method
input_method = st.radio(
    "Choose input method:",
    ("Enter URL", "Upload PDF file"),
    index=0 ,# Default to URL input,
    on_change=clear_history_callback
)

##################################################### Session variables#########################################


if 'message_history' not in st.session_state:
     st.session_state['message_history']=[]
     print ('inside')
if 'url_uploaded' not in st.session_state:
     st.session_state['url_uploaded']=0
if 'file_path' not in st.session_state:
     st.session_state['file_path']=''
if "url_uploader_key" not in st.session_state:
        st.session_state["url_uploader_key"] = ''
if "url_processed" not in st.session_state:
        st.session_state["url_processed"] = ''
if "url_done" not in st.session_state:
        st.session_state["url_done"] = ''


if 'pdf_uploaded' not in st.session_state:
     st.session_state['pdf_uploaded']=0
if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
def reset_session_state():
    # Delete all the keys in session state
    for key in st.session_state.keys():
        if key != "file_uploader_key":
            del st.session_state[key]
    
    st.session_state["file_uploader_key"] += 1

with st.sidebar:
     
          #st.text_input("URL uploaded is :",key=st.session_state["url_done"], value= st.session_state["url_processed"], disabled=True)
          #url_list=URL_loader(url)
     st.markdown('''
        ### About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)
        - [Azure OpenAI Tutorial](https://techcommunity.microsoft.com/t5/startups-at-microsoft/build-a-chatbot-to-query-your-documentation-using-langchain-and/ba-p/3833134)
        - [Git Hub](https://github.com/dharmsingh4u/Study)
        ''')
     st.write("Made ❤️ by Dharmendra")

# 2. Conditionally display the relevant input widget
if input_method == "Enter URL":
    url = st.text_input("Please enter the URL:")
    if url:
        st.write(f"You entered URL: {url}")
        # Add your URL processing logic here
        # Example: Fetch content
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                st.success("Successfully fetched URL content!")
                # Process the content as needed
                st.session_state["url_uploader_key"]=url
                st.session_state["url_processed"]=url
            else:
                st.error("Failed to fetch content from the URL.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

elif input_method == "Upload PDF file":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        full_path = tmp_file.name
                        st.session_state['pdf_uploaded']=1
                        st.session_state['file_path']=full_path
        # Add your PDF processing logic here
        # Example: Read PDF content (requires pdfplumber)
        # try:
        #     with pdfplumber.open(uploaded_file) as pdf:
        #         page = pdf.pages[0] # Get the first page
        #         st.write("Content of the first page:")
        #         st.write(page.extract_text()[:500] + "...") # Display first 500 chars
        # except Exception as e:
        #     st.error(f"Error processing PDF: {e}")

# Note: For more complex apps, consider using Streamlit's 
# session state to maintain data across reruns.


model=ChatOpenAI(model='gpt-4o',temperature=0, max_completion_tokens=2000)
human=HumanMessagePromptTemplate.from_template('Please answer the question {question} in the given context {context} and say no if you dont any ' \
'relevant data from it ')
promt_pdf=ChatPromptTemplate.from_messages([human])
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
input = st.chat_input("What would you like to ask")
if input and  st.session_state["url_uploader_key"]:
     
     prompt=ChatPromptTemplate.from_messages([('system','You are a Education education counsellor who helps students to resolve queries'),('human','Answer this quesiton {question} from the provided context only and here is the context {context} and if you dont the please say dont know')])
     st.session_state['message_history'].append({'role':'user','content':input})
     #chat_history.append(HumanMessage(content=input))
     with st.chat_message('user'):
        st.text(input)
     url_list=URL_selector(model,input,st.session_state["url_uploader_key"])
     retriever=URL_loader(url_list)
     parallel_chain=RunnableParallel({'question':RunnablePassthrough(),'context':retriever|retriver_questions})
     parser=StrOutputParser()
     final_chain=parallel_chain | prompt|model|parser
     result=final_chain.invoke(input)
     #result=final_chain.invoke({'input':input,'chat_history':chat_history})
     st.session_state['message_history'].append({'role':'assitant','content':result})
     #chat_history.append(AIMessage(content=result))
     with st.chat_message('assitant'):

        st.text(result)
elif input and st.session_state['pdf_uploaded']==1:
     
     st.session_state['message_history'].append({'role':'user','content':input})
     with st.chat_message('user'):
        st.text(input)
     full_path=st.session_state['file_path']
     retriver =loader(full_path)
     parallel_chain = RunnableParallel({
     'context': retriver | RunnableLambda(format_docs),
     'question': RunnablePassthrough()
     })
     parser=StrOutputParser()
     main_chain = parallel_chain | promt_pdf | model | parser
     result =main_chain.invoke(input)
    #result=main_chain.stream(input,stream_mode='messages') ## for handling the stream
    #ai_message=st.write_stream( message_chunk.content for message_chunk, metadata in result)
     st.session_state['message_history'].append({'role':'assitant','content':result})
     with st.chat_message('assitant'):


        st.text(result)
     
     
     

     


