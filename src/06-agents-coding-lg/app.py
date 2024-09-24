import os
import json
import dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
import getpass
import random
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from promptflow.tracing import start_trace

dotenv.load_dotenv()
# start a trace session, and print a url for user to check trace
start_trace()

# enable langchain instrumentation
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
instrumentor = LangchainInstrumentor()
if not instrumentor.is_instrumented_by_opentelemetry:
    instrumentor.instrument()

st.title("💬 AI agentic code reviewer")
st.caption("🚀 A set of agents that can generate, review and execute code")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, ToolMessage):
        with st.chat_message("Tool"):
            st.markdown(message.content)
    else:
        with st.chat_message("Agent"):
            st.markdown(message.content)

model: AzureChatOpenAI = None
embeddings_model: AzureOpenAIEmbeddings = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    model = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )

def llm(x):
    return model.invoke(x).content

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])

from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
import random
from typing import Annotated, Sequence, TypedDict

class GraphState(TypedDict):
    feedback: Optional[str] = None
    history: Optional[str] = None
    code: Optional[str] = None
    specialization: Optional[str]=None
    rating: Optional[str] = None
    iterations: Optional[int]=None
    code_compare: Optional[str]=None
    actual_code: Optional[str]=None

workflow = StateGraph(GraphState)

### Edges

reviewer_start= "You are Code reviewer specialized in {}.\
You need to review the given code following PEP8 guidelines and potential bugs\
and point out issues as bullet list.\
Code:\n {}"

coder_start = "You are a Coder specialized in {}.\
Improve the given code given the following guidelines. Guideline:\n {} \n \
Code:\n {} \n \
Output just the improved code and nothing else."

rating_start = "Rate the skills of the coder on a scale of 10 given the Code review cycle with a short reason.\
Code review:\n {} \n "

code_comparison = "Compare the two code snippets and rate on a scale of 10 to both. Dont output the codes.Revised Code: \n {} \n Actual Code: \n {}"

classify_feedback = "Are all feedback mentioned resolved in the code? Output just Yes or No.\
Code: \n {} \n Feedback: \n {} \n"

### Nodes

def handle_reviewer(state):
    history = state.get('history', '').strip()
    code = state.get('code', '').strip()
    specialization = state.get('specialization','').strip()
    iterations = state.get('iterations')
    
    print("Reviewer working...")
    
    feedback = llm(reviewer_start.format(specialization,code))
    
    return {'history':history+"\n REVIEWER:\n"+feedback,'feedback':feedback,'iterations':iterations+1}

def handle_coder(state):
    history = state.get('history', '').strip()
    feedback = state.get('feedback', '').strip()
    code =  state.get('code','').strip()
    specialization = state.get('specialization','').strip()
    
    print("CODER rewriting...")
    
    code = llm(coder_start.format(specialization,feedback,code))
    return {'history':history+'\n CODER:\n'+code,'code':code}

def handle_result(state):
    print("Review done...")
    
    history = state.get('history', '').strip()
    code1 = state.get('code', '').strip()
    code2 = state.get('actual_code', '').strip()
    rating  = llm(rating_start.format(history))
    
    code_compare = llm(code_comparison.format(code1,code2))
    return {'rating':rating,'code_compare':code_compare}

# Define the nodes we will cycle between
workflow.add_node("handle_reviewer",handle_reviewer)
workflow.add_node("handle_coder",handle_coder)
workflow.add_node("handle_result",handle_result)

def deployment_ready(state):
    deployment_ready = 1 if 'yes' in llm(classify_feedback.format(state.get('code'),state.get('feedback'))) else 0
    total_iterations = 1 if state.get('iterations')>5 else 0
    return "handle_result" if  deployment_ready or total_iterations else "handle_coder" 


workflow.add_conditional_edges(
    "handle_reviewer",
    deployment_ready,
    {
        "handle_result": "handle_result",
        "handle_coder": "handle_coder"
    }
)

workflow.set_entry_point("handle_reviewer")
workflow.add_edge('handle_coder', "handle_reviewer")
workflow.add_edge('handle_result', END)

# Compile

specialization = 'python'
problem = 'Generate code to train a Regression ML model using a tabular dataset following required preprocessing steps.'
code = llm(problem)

app = workflow.compile()
conversation = app.invoke({"history":code,"code":code,'actual_code':code,"specialization":specialization,'iterations':0},{"recursion_limit":100})

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    inputs = {
        "messages": [
            ("user", human_query),
        ]
    }

    with st.chat_message("Human"):
        st.markdown(human_query)

    for event in app.stream(inputs):       
        print ("message: ")
        for value in event.values():
            print(value)
            if ( isinstance(value["messages"][-1], AIMessage) ):
                print("AI:", value["messages"][-1].content)
                with st.chat_message("Agent"):
                    if (value["messages"][-1].content == ''):
                        toolusage = ''
                        for tool in value["messages"][-1].additional_kwargs["tool_calls"]:
                            print(tool)
                            toolusage += "id:" + str(tool["index"]) + "  \n name: " + tool["function"]["name"] + "  \n arguments: " + tool["function"]["arguments"] + "  \n\n"
                        st.write("Using the folllwing tools: \n", toolusage)
                    else:
                        st.write(value["messages"][-1].content)
            
            if ( isinstance(value["messages"][-1], ToolMessage) ):
                print("Tool:", value["messages"][-1].content)
                with st.chat_message("Tool"):
                    st.write(value["messages"][-1].content.replace('\n\n', ''))
            
            if ( isinstance(value["messages"][-1], str) ):
                print("Agent:", value["messages"][-1])
                with st.chat_message("AI"):
                    st.write(value["messages"][-1])
