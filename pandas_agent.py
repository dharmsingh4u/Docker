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
sys.path.insert(1, r'D:\Notebooks\LLM\env')
#sys.path.insert(2, r'D:\Notebooks\LLM\langchain_document_loader')
from enviorment import load_env
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
#from pydirectoryloader import rag_function
import os 
load_env()
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import HumanMessagePromptTemplate
import requests
from bs4 import BeautifulSoup

from urllib.parse import urljoin, urlparse
from tavily import TavilyClient
from langchain_community.tools import DuckDuckGoSearchRun
import time
import yfinance as yf
from langchain_core.tools import tool
from ta.momentum import RSIIndicator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

system_prompt ="""You are an expert business data analyst responsible for explaining data to a CEO with no technical background.  
Your job is to analyze the dataframe and produce clear, concise, and valuable business insights.

Follow these rules:

### 1. Communication Style
- Use simple business language.
- No technical terms such as dataframe, index, NaN, syntax, lists, or code.
- Focus on what matters to a CEO: performance, trends, risks, opportunities, decisions.

### 2. Smart Narrative Generation
Whenever the user asks for insights or explanations, automatically generate:

A. Key Highlights  
- What stands out the most?  
- Any strong increases, declines, or patterns?

B. Trends  
- Year-over-year or month-over-month movement  
- Positive or negative direction  
- Growth vs. slowdown

C. Exceptions & Anomalies  
- Unusual spikes  
- Underperforming segments  
- Sudden changes

D. Drivers & Root Causes  
- Why are changes happening?  
- What variables are contributing?

E. Risks & Opportunities  
- Identify potential red flags  
- Highlight upside potential

F. Recommendations  
- Clear, action-oriented advice for the CEO  
- Keep it strategic, not technical

### 3. Graphs & Visualization
- If the CEO requests charts, generate them using matplotlib.  
- Use titles, labels, and clean formatting.  
- Explain what the chart means in simple language.

### 4. Dataset Awareness
You already have access to the dataframe.  
Understand the columns, values, patterns, correlations, and distributions.  
Use this knowledge to generate insightful narratives.

### 5. Never Show Code
Do not show Python, pandas, or any technical implementation details.

### 6. Always Be Insight-Driven
Your goal: Help the CEO make decisions, identify risks, and understand business performance.
Keep your answers high-level, strategic, and insightful.
"""
st.title("📊 CEO Data Intelligence Assistant")
st.write("Upload a CSV/Excel and ask anything — the assistant will decide whether to give insights, summaries, or graphs.")
def create_data_dictionary(df):

    dictionary = []
    for col in df.columns:
        dictionary.append({
            "column": col,
            "type": str(df[col].dtype),
            "example": df[col].dropna().iloc[0] if df[col].dropna().size > 0 else None,
            "missing_values": df[col].isna().sum()
        })
    return dictionary
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
if uploaded_file:
    # Load dataframe
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("Dataset uploaded successfully!")
    data_dict = create_data_dictionary(df)
    prompt = system_prompt + f"""

        ### DATA DICTIONARY (Automatically included):
        {data_dict}

        Use this to understand the dataset better and generate more accurate insights.
        """
model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",   # or gemini-2.0-pro, 1.5-pro, etc.
    temperature=0.2
)
import re
with st.chat_message('user'):
    st.text('Hi')

with st.chat_message('assistant'):
    st.text('how i can help you..')
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
text_input =st.chat_input('Please type here..')
if text_input :
    st.session_state['message_history'].append({'role':'assitant','content':text_input})
    with st.chat_message('user'):
        st.text(text_input)
    agent = create_pandas_dataframe_agent(
    model,
    df,allow_dangerous_code=True,system=system_prompt,verbose=True, return_intermediate_steps=True
)
    response=agent.invoke(text_input)
    with st.chat_message('assitant'):

        st.text(response)
    st.session_state['message_history'].append({'role':'assitant','content':response})
    # if "```python" in response:
    #     print ('inside the function,',response)

    #     #narrative = response.split("```python")[0].strip()
    #     code = response.split("```python")[1].split("```")[0]
    # else:
    #     code = None
#     pattern = r"Action Input:(.*?)<string>:"  # everything between Action Input: and <string>:
#     match = re.search(r"(import matplotlib.*?plt\.show\(\))", response, re.DOTALL)
# #st.write(narrative)  # Show narrative
#     print ('match is ',match)
    intermediate_steps = response['intermediate_steps']  # list of (action, observation) tuples
    python_codes = []
    for action, observation in intermediate_steps:
        # Only capture Python REPL / AST steps
        if action.tool.lower() in ["python_repl_ast", "python"]:
            code = action.tool_input
            code=code.replace("\\n", "\n").strip()
            python_codes.append(code)
    for i, code in enumerate(python_codes, 1):
        print(f"\n--- Python code block #{i} ---\n")
        code=code
    print ('python_codes ',python_codes)
    if code:
        python_code = code
        fig, ax = plt.subplots()
        local_env = {"df": df, "plt": plt, "ax": ax}
        exec(python_code, local_env)

        # Capture chart image
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        chart_image = buf.getvalue()

        # Display chart
        with st.chat_message('assitant'):
            
            st.image(chart_image)

    #message =message

   

