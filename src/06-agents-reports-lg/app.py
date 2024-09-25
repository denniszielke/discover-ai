import os
import dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
import random
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Sequence, TypedDict, Optional
from bs4 import BeautifulSoup
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import List
from pydantic import BaseModel, Field

dotenv.load_dotenv()

st.set_page_config(
    page_title="AI financial analyst agents",
)

st.title("💬 Agentic Finanical Analyst")
st.caption("🚀 A set of financial agents that can generate, validate and iterate on financial statements")

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

chat_model: AzureChatOpenAI = None
model: AzureOpenAI = None
embeddings_model: AzureOpenAIEmbeddings = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    model = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = "2024-08-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    chat_model = AzureChatOpenAI(
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
    model = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version = "2024-08-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    chat_model = AzureChatOpenAI(
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
    return chat_model.invoke(x).content

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

class Objective(BaseModel):
    urls: List[str]
    question: str

def model_objective(input) -> Objective:
    completion = model.beta.chat.completions.parse(
        model = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        messages = [{"role" : "assistant", "content" : f"""Extract all the urls in the following input and the objective that was asked in the form of a question. Input: {input}"""}],
        response_format = Objective)
    
    print(completion)
    return completion.choices[0].message.parsed


def load_financial_report(url: Annotated[str, "Full qualified url of the report to download. Example: https://annualreport2023.volkswagen-group.com/divisions/volkswagen-financial-services.html"]) -> str:
    """This tool loads financial reports from the web and returns the content"""

    doc = WebBaseLoader(url).load()[0]
    print(doc)

    content = "Reference: " + doc.metadata["title"] + " URL: " + url + "content: " + doc.page_content
    return content

def prepare_flow(input:str) -> TypedDict:

    objective = model_objective(input)

    reports = ""

    for url in objective.urls:
        reports += load_financial_report(url)

    inputs = {
        "history": "",
        "insights":reports,
        "statements": "",
        'original_statements':"",
        "specialization":objective.question,
        'iterations':0}
    return inputs

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])

qdrant_client = QdrantClient(":memory:")

qdrant_client.create_collection(
    collection_name="reports",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="reports",
    embedding=embeddings_model,
)

retriever = vector_store.as_retriever()



class ReportState(TypedDict):
    feedback: Optional[str] = None
    history: Optional[str] = None
    statements: Optional[str] = None
    specialization: Optional[str] = None
    insights: Optional[str] = None
    rating: Optional[str] = None
    iterations: Optional[int]=None
    statements_compare: Optional[str] = None
    original_statements: Optional[str] = None
    messages: Annotated[Sequence[BaseMessage], add_messages] = []

workflow = StateGraph(ReportState)

### Nodes

reviewer_start= "You are a senior financial analyist with extensive experience in comparing financial reports with a special attentions to {}.\
You need to review the financial analyst statements, analyse the insights available and the conclusions made from these statements.\
Put a special focus on the insights and the statements under the given specialization and based on the feedback, provide a revised version of the statements.\
Make sure that url references are provided for the major statements. \
Make sure that you highlight the key points of your thinking in the reasoning output. \
Insights:\n {} \
Statements: \n {}"

def handle_reviewer(state):
    print("starting reviewer")
    print(state)
    history = state.get('history', '').strip()
    statements = state.get('statements', '').strip()
    insights = state.get('insights', '').strip()
    specialization = state.get('specialization','').strip()
    iterations = state.get('iterations')
    messages = state.get('messages')
    print("reviewer working...")
    
    feedback = model_response(reviewer_start.format(specialization,insights,statements))
    if (feedback.certainty > 0.85):
        messages.append(AIMessage(content="Reviewer (" + str(feedback.certainty) +  "): "+feedback.reasoning))
    print("reviewer done")
    print(feedback)

    return {'history':history+"\n REVIEWER:\n"+feedback.response,'feedback':feedback.response,'iterations':iterations+1, 'insights': insights, 'messages':messages}

analyst_start = "You are an financial consultant specialized in {}.\
Analyse the financial statements of the company and provide valuable insights for market opportunities and unique challenges.\
Financial performance analysis is the process of assessing a company’s financial health and making informed decisions by analyzing key metrics and techniques. It involves reviewing financial statements, such as the balance sheet, income statement, cash flow statement, and annual report, to gain insights into profitability, liquidity, solvency, efficiency, and valuation.\n \
Financial KPIs, or key performance indicators, are metrics used to track, measure, and analyze the financial health of a company. These KPIs fall under categories like profitability, liquidity, efficiency, solvency, and valuation. They include metrics such as gross profit margin, current ratio, inventory turnover, debt-to-equity ratio, and price-to-earnings ratio.\
Improve the given statements given this feedback and the available raw insights. Make sure you name the url for reference for the major statements. Feedback:\n {} \n \
Statements:\n {} \n \
Insights:\n {} \n \
Output just the revised statements and add nothing else. Make sure that you document your reasoning for the major statements in the output."

def handle_analyst(state):
    print("starting analyst")
    print(state)
    history = state.get('history', '').strip()
    feedback = state.get('feedback', '').strip()
    insights = state.get('insights', '').strip()
    statements =  state.get('statements','').strip()
    specialization = state.get('specialization','').strip()
    messages = state.get('messages')
    print("analyst rewriting...")
    
    analsis = model_response(analyst_start.format(specialization,feedback,statements,insights))
    messages.append(AIMessage(content="Analyst (" + str(analsis.certainty) +  "): "+analsis.reasoning))
    messages.append(SystemMessage(content=statements))

    print("analyst done")
    return {'history':history+'\n STATEMENTS:\n'+analsis.response,'statements':analsis.response, 'insights': insights, 'messages':messages}

statement_comparison = "Compare the two statements and rate on a scale of 10 to both. Dont output the statements.Revised statements: \n {} \n Original statements: \n {}"

rating_start = "Rate the skills of the financial insights on a scale of 10 given the statement review cycle with a short reason.\
Statement review:\n {} \n "

def handle_result(state):
    print("Review done...")
    
    history = state.get('history', '').strip()
    code1 = state.get('statements', '').strip()
    code2 = state.get('original_statements', '').strip()
    messages = state.get('messages')
    rating  = model_response(rating_start.format(history))
    
    messages.append(AIMessage(content="Rating (" + str(rating.certainty) +  "): "+rating.reasoning))

    statements_compare = llm(statement_comparison.format(code1,code2))

    messages.append(AIMessage(content="Result: "+statements_compare))

    messages.append(SystemMessage(content=code1))
    return {'rating':rating,'code_compare':statements_compare, 'messages':messages}

# Define the nodes we will cycle between
workflow.add_node("handle_reviewer",handle_reviewer)
workflow.add_node("handle_analyst",handle_analyst)
workflow.add_node("handle_result",handle_result)


classify_feedback = "Are most of the important feedback points mentioned resolved in the statements? Output just Yes or No.\
Statements: \n {} \n Feedback: \n {} \n"

def deployment_ready(state):
    deployment_ready = 1 if 'yes' in llm(classify_feedback.format(state.get('statements'),state.get('feedback'))) else 0
    total_iterations = 1 if state.get('iterations')>5 else 0
    print(state);
    if state.get('iterations')>8:
        print("Iterations exceeded")
        return "handle_result"
    return "handle_result" if  deployment_ready or total_iterations else "handle_analyst" 


workflow.add_conditional_edges(
    "handle_reviewer",
    deployment_ready,
    {
        "handle_result": "handle_result",
        "handle_analyst": "handle_analyst"
    }
)

workflow.set_entry_point("handle_analyst")
workflow.add_edge('handle_analyst', "handle_reviewer")
workflow.add_edge('handle_result', END)

# Compile

app = workflow.compile()

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    inputs = prepare_flow(human_query)
    config = {"recursion_limit":20}

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
        
