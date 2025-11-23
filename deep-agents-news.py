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
import dotenv
dotenv.load_dotenv()
#sys.path.insert(1, r'D:\Notebooks\LLM\env')
#sys.path.insert(2, r'D:\Notebooks\LLM\langchain_document_loader')
#from enviorment import load_env
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
#from pydirectoryloader import rag_function
import os 
#load_env()
#TAVILY_API_KEY=os.getenv('TAVILY_API_KEY')
#print ('api key ',TAVILY_API_KEY)
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
import asyncio
from langchain_core.tools import tool
from ta.momentum import RSIIndicator
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
from langchain_google_genai import ChatGoogleGenerativeAI
#model=ChatOpenAI()
model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",   # or gemini-2.0-pro, 1.5-pro, etc.
    temperature=0.2
)
#model =ChatOpenAI(temperature=0,max_completion_tokens=1000,model='gpt-4')
model=model
#mcp_client = MultiServerMCPClient(r'D:\Notebooks\Langraph\MCP\browser.json')
from deepagents import create_deep_agent
import streamlit as st
#model =ChatOpenAI(temperature=0,max_completion_tokens=1000,model='gpt-4')
st.title('Gotta Go, Go, Go')
st.set_page_config(
        page_title="QnA",
)
##########################################################Streamlit##################################################
if 'message_history' not in st.session_state:
     st.session_state['message_history']=[]

# ------------------------------------------------------ SIDEBAR ------------------------------------------------------ #

with st.sidebar:

       
                     
                    
                    
                         
                        

        

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
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
input = st.chat_input("Enter your City")
prompt="""You are a professional travel assistant. Your goal is to provide comprehensive, accurate, and user-friendly information to tourists about any city they ask about. Follow these rules:

1. **City Overview:** Start with a brief introduction about the city, including highlights, key attractions, and what the city is known for.

2. **Recommended Places to Visit:** Suggest must-see locations, attractions, landmarks, restaurants, or local experiences suitable for tourists. Include:
   - Name of the place
   - Short description
   - Approximate opening hours or best time to visit (if known)

3. **Current News:** Provide a brief summary of any recent news or notable events in the city that a tourist might find relevant.

4. **Crime & Safety:** Include information about crime trends, safety alerts, or areas to be cautious about, presented clearly and without causing unnecessary alarm.

5. **Weather:** Give the current weather conditions and a short-term forecast (1–3 days), including temperature, conditions (sunny, rainy, etc.), and any weather warnings.

6. **Travel Tips:** Include helpful suggestions for tourists, such as transportation options, cultural etiquette, or seasonal advice.

7. **Hotels:** Include helpful Hotel suggestion by using MCP tools using Airbnb .

8. **Format:** Present the answer in a clear, structured format. Example:

City: [CITY_NAME]  
Date: [CURRENT_DATE]

**Overview**  
- [Brief introduction]

**Recommended Places to Visit**  
- [Place 1]: [Description]  
- [Place 2]: [Description]

**Current News**  
- [Headline / Summary] ([Date / Source])

**Crime & Safety**  
- [Summary / Trends / Alerts]

**Weather**  
- Current: [Temperature], [Conditions]  
- Forecast: [Next 1–3 days]

**Travel Tips**  
- [Tip 1]  
- [Tip 2]

**Hotels**  
- [Hotel 1]  
- [Hotel 2]

8. **Tone:** Friendly, helpful, and professional. Avoid speculation. If certain information is unavailable, clearly mention it.

Whenever a tourist asks about a city, follow this structure and provide all available information, drawing from MCP tools for news, weather, crime,Hotel, and local places.

"""

def getweahter(city:str)->dict:
     """"this function is to get weather report for a given city anyhwere in the world"""
     url = f'https://api.weatherstack.com/forecast?access_key=ec07421f5107d04ec9fa783c8fcfd6c5&query={city}'
     response = requests.get(url)
     return response.json()

async def main():
    #print('before tools')
    with open(r'client.json', "r") as f:
        data = json.load(f)
        #data=data.replace("${TAVILY_API_KEY}", TAVILY_API_KEY)
    servers = data["mcpServers"]
    #servers["tavily-remote-mcp"]["url"]
    #print ('servers ',servers)
# Convert dict → positional server list
    # server_list = []
    # for name, cfg in servers.items():
    #     cfg["name"] = name   # MultiServerMCPClient requires each server to have 'name'
    #     server_list.append(cfg)
    # print('server_list .',server_list)
    mcp_client =  MultiServerMCPClient(servers)
    mcp_tools = await mcp_client.get_tools()
    #print('before tools 2')
    his=[getweahter]
    tools=his+ mcp_tools
    tools= mcp_tools
    agent = create_deep_agent(tools=tools,model=model,system_prompt=prompt)
    #print('after tools')
    #='Please compare NVIDIA and TSLA stock and give your insights'
    async for chunk in agent.astream({"messages": [{"role": "user", "content": input}]}):
            
            if "messages" in chunk:
                for msg in chunk["messages"]:
                # Check if it's an AI message
                    if getattr(msg, "type", None) == "AIMessage":
                        if hasattr(msg, "pretty_print"):
                            pass
                            #msg.pretty_print()
                        else:
                            pass
                            #print(msg.get("content", msg), end="")

        # 2️⃣ If the chunk has 'text' (common for AI output)
            elif "model" in chunk:
                #pass
                for i in chunk['model']['messages']:

                    with st.chat_message('assitant'):
                        print('content is ',i.content)
                        if "#" not in i.content:

                            print('content inside is',i.content) 
                            st.text(i.content)
                            st.session_state['message_history'].append({'role':'assitant','content':i.content})

            # 3️⃣ If the chunk has 'output_text' (tool output — skip)
            # Ignore tool outputs for AI-only view

            # 4️⃣ Fallback — ignore everything else
            else:
                print(chunk)
                pass
if input:
      with st.chat_message('user'):
        st.text(input)
      st.session_state['message_history'].append({'role':'user','content':input})

      asyncio.run(main())
