import os
import dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores import AzureSearch
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from promptflow.tracing import start_trace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from langchain.globals import set_verbose

set_verbose(True)

dotenv.load_dotenv()
# start a trace session, and print a url for user to check trace
start_trace()

# enable langchain instrumentation
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
instrumentor = LangchainInstrumentor()
if not instrumentor.is_instrumented_by_opentelemetry:
    instrumentor.instrument()

    
st.title("💬 AI bot that can use AI Search")
st.caption("🚀 A Bot that can use a vector store to answer questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

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

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_response(query, chat_history):
    template = """Use the given context to answer the question. 
    If you don't know the answer, say you don't know. 
    Use three sentence maximum and keep the answer concise. 

    Chat history:
    {chat_history}

    Context: 
    {context}
    
    User question:
    {question}
    
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(),  "chat_history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream(
        {"input": query, "chat_history": chat_history}
    )

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    with st.chat_message("Human"):
        st.markdown(human_query)
    with st.chat_message("assistant"):
        ai_response = st.write_stream(get_response(human_query, st.session_state.chat_history))

    st.session_state["chat_history"].append(AIMessage(ai_response))
