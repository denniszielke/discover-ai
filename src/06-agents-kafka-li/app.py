import os
import json
import dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from llama_agents.message_queues.apache_kafka import KafkaMessageQueue

from llama_agents import (
    AgentService,
    HumanService,
    ControlPlaneServer,
    SimpleMessageQueue,
    PipelineOrchestrator,
    ServiceComponent,
    LocalLauncher,
)
from llama_agents import (
    AgentService,
    AgentOrchestrator,
    ControlPlaneServer,
    SimpleMessageQueue,
)

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.core.query_pipeline import RouterComponent, QueryPipeline
from llama_index.llms.openai import OpenAI
from llama_index.core.selectors import PydanticSingleSelector

dotenv.load_dotenv()


from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.WARNING
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import Settings

KAFKA_CONNECTION_URL = os.getenv("KAFKA_URL")

llm: AzureOpenAI = None
embeddings_model: AzureOpenAIEmbedding = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureOpenAI(
        model=os.getenv("AZURE_OPENAI_COMPLETION_MODEL"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_VERSION")
    )

    embeddings_model = AzureOpenAIEmbedding(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    llm = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        use_azure_ad=True,
        model=os.getenv("AZURE_OPENAI_COMPLETION_MODEL"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        api_key=token_provider(),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_VERSION")
    )
    embeddings_model = AzureOpenAIEmbedding(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
    )

Settings.llm = llm
Settings.embed_model = embeddings_model

# # create our multi-agent framework components
# message_queue = SimpleMessageQueue()
from llama_index.llms.ollama import Ollama

phi = Ollama(base_url='http://localhost:11434', model='phi3.5', temperature=0.8, request_timeout=300,
             system_prompt="You are an agent that returns a random response to any question. You change the topic and answer the negative of the corr")

import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    WARNING = '\033[92m'
    ENDC = '\033[0m'

# create an agent
def get_machine_status() -> str:
    """Returns information about the production machine"""
    print(f"{bcolors.WARNING}#### Returning machine status from tool...{bcolors.ENDC}")
    time.sleep(10)
    print(f"{bcolors.WARNING}#### Returned machine status from tool{bcolors.ENDC}")
    return "The machine is healthy."
def get_number_of_machine_jobs() -> str:
    """Returns information about the number of jobs on the production machine"""
    print(f"{bcolors.WARNING}#### Returning number of machine jobs from tool...{bcolors.ENDC}")
    time.sleep(10)
    print(f"{bcolors.WARNING}#### Returned number of machine jobs from too{bcolors.ENDC}")
    return "The machine has 5 jobs."

def get_order_status() -> str:
    """Returns information about the current order status"""
    print(f"{bcolors.WARNING}#### Returning order status from tool...{bcolors.ENDC}")
    time.sleep(10)
    print(f"{bcolors.WARNING}#### Returned order status from tool{bcolors.ENDC}")
    return "There are two new orders in the system."

machine_tools_1 = FunctionTool.from_defaults(fn=get_machine_status)
machine_tools_2 = FunctionTool.from_defaults(fn=get_number_of_machine_jobs)
order_tools_1 = FunctionTool.from_defaults(fn=get_order_status)

agent1 = ReActAgent.from_tools([machine_tools_1, machine_tools_2], llm=llm)
agent2 = ReActAgent.from_tools([], llm=llm)

from llama_index.core.tools import FunctionTool

message_queue = KafkaMessageQueue(url=KAFKA_CONNECTION_URL)

# create our multi-agent framework components
control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=llm),
    port=8001,
)
agent_server_1 = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for getting the information and status of the production machine.",
    service_name="machine_information_agent",
    port=8002,
)
agent_server_2 = AgentService(
    agent=agent2,
    message_queue=message_queue,
    description="Useful for getting information about order status.",
    service_name="order_status_agent",
    port=8003,
)

from llama_agents import LocalLauncher
import nest_asyncio

# needed for running in a notebook
nest_asyncio.apply()

# launch it
launcher = LocalLauncher(
    [agent_server_1, agent_server_2],
    control_plane,
    message_queue,
)

while True:
    try:
        line = input()
        print(f"{bcolors.OKBLUE}#### Question: {line}{bcolors.ENDC}")
        result = launcher.launch_single(line)
        print(f"{bcolors.WARNING}#### Response: {result}{bcolors.ENDC}")
    except EOFError:
        break
    if line == "q":
        break