import os
import pytz
from datetime import datetime
import dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_react_agent
from langchain.globals import set_verbose
from langchain_core.tools import tool

from langchain import agents
from langchain_core.prompts import PromptTemplate

from langchain_core.tools import tool

from promptflow.tracing import start_trace
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    VectorizedQuery
)
from typing import Annotated, List

set_verbose(True)

dotenv.load_dotenv()

# enable langchain instrumentation
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
instrumentor = LangchainInstrumentor()
if not instrumentor.is_instrumented_by_opentelemetry:
    instrumentor.instrument()

# start a trace session, and print a url for user to check trace
start_trace()

credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_KEY"]) if len(os.environ["AZURE_AI_SEARCH_KEY"]) > 0 else DefaultAzureCredential()
index_name = "products-semantic-index"
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
 
client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

search_client = SearchClient(
    endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"], 
    index_name=index_name,
    credential=credential
)

st.set_page_config(
    page_title="AI bot that can use multiple tools"
)

st.title("💬 AI bot that can use tools in combination")
st.caption("🚀 A Bot that can use a vector store, humans and tools to answer questions")

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

@tool()
def create_order(user_id: Annotated[int, "the user id, which is int. Example 105"], product_list: Annotated[List[str], "Product list as strings"]) -> str:
    'This tool creates order, it is expecting an object, the object has 2 properties. First one is named user_id, which is a number. Second one is named product_list, which is a list of strings. It returns a string.'
    print("Creating order for user_id: ", user_id)
    print("Order details: ", product_list)
    return "Success"

@tool
def get_current_location(input: str) -> str:
    "Get the current timezone location of the user."
    return "Europe/Berlin"

@tool
def get_current_time(location: str) -> str:
    "Get the current time in the given location. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London. Anything in Germany should be Europe/Berlin"
    try:
        print("get current time for location: ", location)
        timezone = pytz.timezone(location)
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")
        return current_time
    except Exception as e:
        print("Error: ", e)
        return "Sorry, I couldn't find the timezone for that location."

@tool
def sum(a: int, b: int) -> int:
    "Sum two numbers. its a math tool."
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    "Multiply two numbers. its a math tool."
    return a * b

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# use an embeddingsmodel to create embeddings
def get_embedding(text, model=embedding_model):
    if len(text) == 0:
        return client.embeddings.create(input = "no description", model=model).data[0].embedding
    return client.embeddings.create(input = [text], model=model).data[0].embedding

@tool
def search_for_product(question: str) -> str:
    """This will return more detailed information about the products from the product repository Returns top 5 results."""
    # create a vectorized query based on the question
    vector = VectorizedQuery(vector=get_embedding(question), k_nearest_neighbors=5, fields="variantVector")

    found_docs = list(search_client.search(
        search_text=None,
        query_type="semantic", query_answer="extractive",
        query_answer_threshold=0.8,
        semantic_configuration_name="products-semantic-config",
        vector_queries=[vector],
        select=["id", "name", "description", "category", "brand", "price", "tags"],
        top=12
    ))

    print(found_docs)
    found_docs_as_text = " "
    for doc in found_docs:   
        print(doc) 
        found_docs_as_text += " "+ "Name: {}".format(doc["name"]) +" "+ "Description: {}".format(doc["description"]) +" "+ "Brand: {}".format(doc["brand"]) +" "+ "Price: {}".format(doc["price"]) +" "+ "Tags: {}".format(doc["tags"]) +" "+ "Category: {}".format(doc["category"]) +" "

    return found_docs_as_text

@tool
def get_last_purchases(user_id: Annotated[int, "the user id, which is int. Example 105"]) -> List[str]:
    "Returns last 5 purchases of a customer, as List of strings. Call this tool with the user_id which is only a number."
    print("Getting last purchases for userId: ", user_id)
    return ["Lego City Police Station","Ultra-Thin Mechanical Keyboard"]

@tool
def get_user_info(jwtToken: str) -> str:
    "Returns current user/customers information. Name, Address is returned seperated by semi-colon. This function needs the current jwt from the user as parameter."
    return "Name: Dennis; Address: Microsoft Street 1."

@tool
def get_user_id(jwtToken: str) -> int:
    "Returns current user/customers user_id. This function needs the current jwt from the user as parameter."
    return 1234

@tool
def get_jwt() -> int:
    "Returns current user/customers jwt."
    return " ABC1345"

@tool
def get_input(message_to_human: str) -> str:
    'Tool for asking the human for additional information. Please formulate the question in the parameter message_to_human in a form to the human so it can be answered as short as possible.'
    print("Message to human: " + message_to_human)
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)

tools = [get_jwt, get_user_id, get_user_info, search_for_product, get_last_purchases, create_order, sum, multiply, get_input, get_current_time]

prompt_template_v1 = """\
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are talking to customers. Users and customers are the same thing for you.

Assistant is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 

TOOLS:

------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

```

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
"""


prompt_template_v2 = """\
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are talking to customers. Users and customers are the same thing for you.

Assistant is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 

TOOLS:

------

Assistant has access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]
Valid "action" values: "Final Answer" or {tool_names}
```
$JSON_BLOB
```

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

```

If you are about to create an order ask the human for approval.

```

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)
"""
prompt = PromptTemplate.from_template(prompt_template_v2)
agent = create_structured_chat_agent(llm, tools, prompt)

  
agent_executor = agents.AgentExecutor(
        name="Tools Agent",
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10, return_intermediate_steps=True, 
        # handle errors
        error_message="I'm sorry, I couldn't understand that. Please try again.",
    )
 

human_query = st.chat_input()

if human_query is not None and human_query != "":

    st.session_state.chat_history.append(HumanMessage(human_query))

    with st.chat_message("Human"):
        st.markdown(human_query)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": human_query, "chat_history": st.session_state.chat_history}, {"callbacks": [st_callback]}, 
        )

        ai_response = st.write(response["output"])

    # st.session_state["chat_history"].append(AIMessage(ai_response))
