import os
import dotenv
import asyncio
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
import random
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Sequence, TypedDict, Optional
from typing import List

import pytz
from datetime import datetime
from typing_extensions import Annotated


dotenv.load_dotenv()

st.set_page_config(
    page_title="Collaborate with agentic agents and tools",
)

st.title("💬 Agentic Tools Usage")
st.caption("🚀 A set of agents that can use tools")

aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_key = os.getenv("AZURE_OPENAI_API_KEY")
aoai_dalee = os.getenv("AZURE_OPENAI_DALLEE")
aoai_completion = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")

config_list = [
    {
        'model' : aoai_completion, # Your GPT4o deployment name 
        'azure_endpoint' : aoai_endpoint,
        'api_key': aoai_key,
        'api_type' : 'azure',
        'api_version' : '2024-05-01-preview'
    }
]

llm_config={
    "timeout": 600,
    "config_list": config_list,
    "temperature": 0
}
from typing import Annotated, Literal

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")
    
def get_current_location() -> str:
    "Get the current timezone location of the user."
    with st.chat_message("tools"):
        st.write("Retrieving current location")        
    return "Europe/Berlin"

def get_current_time(location: str) -> str:
    "Get the current time in the given location. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London. Anything in Germany should be Europe/Berlin"
    try:
        print("get current time for location: ", location)
        location = str.replace(location, " ", "")
        location = str.replace(location, "\"", "")
        location = str.replace(location, "\n", "")
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")
        with st.chat_message("tools"):
            st.write("Retrieving time for location" + location)   
        return current_time
    except Exception as e:
        print("Error: ", e)
        return "Sorry, I couldn't find the timezone for that location."
    
from autogen import ConversableAgent

# Let's first define the assistant agent that suggests tool calls.
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with simple tasks around time and locations. "
    "Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
)

# The user proxy agent is used for interacting with the assistant agent
# and executes tool calls.
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

# Register the tool signature with the assistant agent.
assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)
assistant.register_for_llm(name="location", description="A tool than can find out the location of the user")(get_current_location)
assistant.register_for_llm(name="time", description="Get the current time in the given location. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London. Anything in Germany should be Europe/Berlin")(get_current_time)

# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="calculator")(calculator)
user_proxy.register_for_execution(name="location")(get_current_location)
user_proxy.register_for_execution(name="time")(get_current_time)

from autogen import register_function

# Register the calculator function to the two agents.
register_function(
    calculator,
    caller=assistant,  # The assistant agent can suggest calls to the calculator.
    executor=user_proxy,  # The user proxy agent can execute the calculator calls.
    name="calculator",  # By default, the function name is used as the tool name.
    description="A simple calculator",  # A description of the tool.
)
human_query = st.chat_input()

if human_query is not None and human_query != "":

    with st.chat_message("Human"):
        st.markdown(human_query)
    
    with st.chat_message("assistant"):
        chat_result = user_proxy.initiate_chat(assistant, message=human_query)
        st.write(chat_result)
        
