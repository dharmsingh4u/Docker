from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
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
#sys.path.insert(2, r'D:\Notebooks\LLM\langchain_document_loader')
#from enviorment import load_env
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
#from pydirectoryloader import rag_function
import os 
load_env()
from pdf_loader import loader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import HumanMessagePromptTemplate
import streamlit as st
import tempfile
st.title('Ask your PDF')
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
model =ChatOpenAI(temperature=0,max_completion_tokens=1000,model='gpt-4')
st.set_page_config(
        page_title="QnA",
)
human=HumanMessagePromptTemplate.from_template('Please answer the question {question} in the given context {context} and say no if you dont any ' \
'relevant data from it ')
promt=ChatPromptTemplate.from_messages([human])

if 'message_history' not in st.session_state:
     st.session_state['message_history']=[]
if 'pdf_uploaded' not in st.session_state:
     st.session_state['pdf_uploaded']=0
if 'file_path' not in st.session_state:
     st.session_state['file_path']=''
if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
def reset_session_state():
    # Delete all the keys in session state
    for key in st.session_state.keys():
        if key != "file_uploader_key":
            del st.session_state[key]
    
    st.session_state["file_uploader_key"] += 1
# ------------------------------------------------------ SIDEBAR ------------------------------------------------------ #
    # Sidebar contents
with st.sidebar:

        # Upload PDF Files
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(label="Upload your PDFs here and click on 'Process'",  type=["pdf"], key=st.session_state["file_uploader_key"])
        process_button = st.button(label="Process")

        if process_button:
            if pdf_docs:
                # st.session_state.clear() # Clear the session state variables
                with st.spinner(text="Processing PDFs..."):
                     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(pdf_docs.name)[1]) as tmp_file:
                        tmp_file.write(pdf_docs.getvalue())
                        full_path = tmp_file.name
                        st.session_state['pdf_uploaded']=1
                        st.session_state['file_path']=full_path
                     
                     
                    
                    
                         
                        

        

        # Web App References
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
        reset = st.sidebar.button('Reset all', on_click=reset_session_state)
        if reset:
            # for key in st.session_state.keys():
            #     if key != "file_uploader_key":
            #         del st.session_state[key]
            # st.session_state["file_uploader_key"] += 1
            # initialize_session_state()
            st.rerun()
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

     
input = st.chat_input("What would you like to know about this document?")
if input and st.session_state['pdf_uploaded']==1:
    print ('input' ,input ,' process_button',process_button)
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
    main_chain = parallel_chain | promt | model | parser
    result =main_chain.invoke(input)
    #result=main_chain.stream(input,stream_mode='messages') ## for handling the stream
    #ai_message=st.write_stream( message_chunk.content for message_chunk, metadata in result)
    st.session_state['message_history'].append({'role':'assitant','content':result})
    with st.chat_message('assitant'):

        st.text(result)




    
    


