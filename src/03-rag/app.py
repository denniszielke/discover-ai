import os
import dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from promptflow.tracing import start_trace
import random

dotenv.load_dotenv()
# start a trace session, and print a url for user to check trace
start_trace()

# enable langchain instrumentation
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
instrumentor = LangchainInstrumentor()
if not instrumentor.is_instrumented_by_opentelemetry:
    instrumentor.instrument()

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

   
st.title("💬 AI bot that can use AI Search")
st.caption("🚀 A Bot that can use a vector store to answer questions")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])

llm: AzureChatOpenAI = None
vector_store: AzureSearch

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
    temperature=0,
    streaming=True
)

retriever = AzureAISearchRetriever(
    content_key="plot", top_k=5, index_name="movies-semantic-index", service_name=os.getenv("AZURE_AI_SEARCH_NAME"), api_key=os.getenv("AZURE_AI_SEARCH_KEY")
)    

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question based only on the following context:\n\n{context}"),
        ("user", "{input}"),
    ]
)

prompt.with_config({"run_name": "ChatPromptTemplate.langchain.task", "run_id": st.session_state["session_id"], "configurable": {"cat": "peter"}, "tags": ["my_template"], "metadata": {"category": "jokes"}})

prompt.output_schema()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

from langchain_core.runnables.config import RunnableConfig

print(chain.Config())

if prompt := st.chat_input():

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        print(response)
        st.write(response)
