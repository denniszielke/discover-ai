import os
import json
import dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
import getpass
import random
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import BaseMessage, SystemMessage
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

class Statement(BaseModel):
    response: str = Field(
        ...,
        description="The response to the question",
    )
    reasoning: str = Field(
        ...,
        description="The reasoning behind the response",
    )
    certainty: float = Field(
        ...,
        description="The certainty of the correctness of the response",
    )

def model_response(input) -> Statement:
    completion = model.beta.chat.completions.parse(
        model = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        messages = [{"role" : "assistant", "content" : f""" Help me understand the following by giving me a response to the question, a short reasoning on why the response is correct and a rating on the certainty on the correctness of the response:  {input}"""}],
        response_format = Statement)
    
    return completion.choices[0].message.parsed

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])

from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import random
from typing import Annotated, Sequence, TypedDict

class GraphState(TypedDict):
    objective: Optional[str] = None
    feedback: Optional[str] = None
    history: Optional[str] = None
    code: Optional[str] = None
    specialization: Optional[str]=None
    rating: Optional[str] = None
    iterations: Optional[int]=None
    code_compare: Optional[str]=None
    actual_code: Optional[str]=None
    messages: Annotated[Sequence[BaseMessage], add_messages] = []

workflow = StateGraph(GraphState)

### Edges







### Nodes

reviewer_start= "You are Code reviewer specialized in {}.\
You need to review the given code following PEP8 guidelines and potential bugs\
and point out issues as bullet list.\
Code:\n {}"

def handle_reviewer(state):
    history = state.get('history', '').strip()
    code = state.get('code', '').strip()
    specialization = state.get('specialization','').strip()
    iterations = state.get('iterations')
    messages = state.get('messages')
    print("Reviewer working...")
    
    feedback = llm(reviewer_start.format(specialization,code))
    messages.append(AIMessage(content="Reviewer: " + feedback))

    return {'history':history+"\n REVIEWER:\n"+feedback,'feedback':feedback,'iterations':iterations+1, 'messages':messages}

coder_start = "You are a Coder specialized in {}.\
Improve the given code given the following guidelines. Guideline:\n {} \n \
Code:\n {} \n \
Output just the improved code and nothing else."
def handle_coder(state):
    history = state.get('history', '').strip()
    feedback = state.get('feedback', '').strip()
    code =  state.get('code','').strip()
    specialization = state.get('specialization','').strip()
    messages = state.get('messages')
    print("CODER rewriting...")
    
    code = llm(coder_start.format(specialization,feedback,code))
    messages.append(AIMessage(content="Coder: " + code))
    return {'history':history+'\n CODER:\n'+code,'code':code, 'messages':messages}

rating_start = "Rate the skills of the coder on a scale of 10 given the Code review cycle with a short reason.\
Code review:\n {} \n "

code_comparison = "Compare the two code snippets and rate on a scale of 10 to both. Dont output the codes.Revised Code: \n {} \n Actual Code: \n {}"

def handle_result(state):
    print("Review done...")
    messages = state.get('messages')
    history = state.get('history', '').strip()
    code1 = state.get('code', '').strip()
    code2 = state.get('actual_code', '').strip()
    rating  = llm(rating_start.format(history))
    
    code_compare = llm(code_comparison.format(code1,code2))
    messages.append(AIMessage(content="Result: " + code_compare))

    return {'rating':rating,'code_compare':code_compare, 'messages':messages}

# Define the nodes we will cycle between
workflow.add_node("handle_reviewer",handle_reviewer)
workflow.add_node("handle_coder",handle_coder)
workflow.add_node("handle_result",handle_result)

classify_feedback = "Are all feedback mentioned resolved in the code? Output just Yes or No.\
Code: \n {} \n Feedback: \n {} \n"
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

app = workflow.compile()
# conversation = app.invoke({"history":code,"code":code,'actual_code':code,"specialization":specialization,'iterations':0},{"recursion_limit":100})

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    specialization = 'python'
    code = llm(human_query)

    inputs = {"objective": human_query, "history":code,"code":code,'actual_code':code,"specialization":specialization,'iterations':0}

    config = {"recursion_limit":100}

    with st.chat_message("Human"):
        st.markdown(human_query)

    for event in app.stream(inputs, config):       
        print ("message: ")
        for value in event.values():
            print("streaming")
            print(value)

            if ( value["messages"].__len__() > 0 ):
                for message in value["messages"]:
                    if (message.content.__len__() > 0):
                        if ( isinstance(message, AIMessage) ):
                            with st.chat_message("AI"):
                                st.write(message.content)
                        elif ( isinstance(message, SystemMessage) ):
                            with st.chat_message("human"):
                                st.write(message.content)
                        else:
                            with st.chat_message("Agent"):
                                st.write(message.content)
