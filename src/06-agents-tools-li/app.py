import os
import json
import dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
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
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import Settings



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

# create our multi-agent framework components
message_queue = SimpleMessageQueue()

# create an agent
def get_the_secret_fact() -> str:
    """Returns the secret fact."""
    return "The secret fact is: A baby llama is called a 'Cria'."

tool = FunctionTool.from_defaults(fn=get_the_secret_fact)

agent1 = ReActAgent.from_tools([tool], llm=llm)
agent2 = ReActAgent.from_tools([], llm=llm)

# create our multi-agent framework components
message_queue = SimpleMessageQueue(port=8000)
control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=llm),
    port=8001,
)
agent_server_1 = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for getting the secret fact.",
    service_name="secret_fact_agent",
    port=8002,
)
agent_server_2 = AgentService(
    agent=agent2,
    message_queue=message_queue,
    description="Useful for getting random dumb facts.",
    service_name="dumb_fact_agent",
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
result = launcher.launch_single("What is the secret fact?")

print(f"Result: {result}")