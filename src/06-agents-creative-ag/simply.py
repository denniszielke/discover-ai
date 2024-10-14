import autogen
import dotenv
import os
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent

dotenv.load_dotenv()

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

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

chief = ConversableAgent(
    "cto",
    system_message="You are Chief technical officer of a tech company responsible for people, strategy and execution",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
)

ai_lead = ConversableAgent(
    "ai_lead",
    system_message="You are innovation officer of a tech company responsible for AI and ML",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
)

assistant = autogen.AssistantAgent(
    name="CTO",
    llm_config=llm_config,
    system_message="Chief technical officer of a tech company"
)

result = chief.initiate_chat(ai_lead, message="Help me come up with a mission statemen for our AI vision.", max_turns=6)

