# Discover Generative AI Apps on Azure
This repository contains demos and templates for building intelligent Apps using OpenAI on Azure.

Regions that this deployment can be executed:
- uksouth
- swedencentral
- canadaeast
- australiaeast


## Inventory

The following scenarios are implemented in this repository:

| Name | Description | Technology  |
| :-- | :--| :-- |
| [Prompt Engineering](./src/01-prompting/basic.ipynb)| Basic prompt engineering examples | Azure Prompt Flow  |
| [Embeddings](./src/02-embeddings/similarity.ipynb)| Learning about embedding models for text and images | Azure OpenAI, Azure Computer Vision  |
| [Vector Search](./src/03-rag/rag-ai-search.ipynb)| Learning about low level retrieval augmented generation | Azure OpenAI, Azure AI Search  |
| [Orchestrated Vector Search](./src/03-rag/app.py)| Learning about retrieval augmented generation with Orchestration | Azure OpenAI, Azure AI Search, LangChain, Streamlit |
| [Persisting Chat Memory](./src/04-orchestration-lc/chat-conversation.ipynb)| Learning about persisting chat messages | Azure OpenAI, Azure AI Search, LangChain, Azure CosmosDB  |
| [Tracing Language Models](./src/04-orchestration-lc/tracing.ipynb)| Learning about tracing language model invocations | Azure OpenAI, LangChain, Azure PromptFlow  |
| [Tracing Chat application](./src/04-orchestration-lc/app.py)| Learning about tracing a running application | Azure OpenAI, LangChain, Azure PromptFlow, Streamlit, Azure AI Search  |
| [Function Calling](./src/05-tools/tools-openai.ipynb)| Learning about using tools from a model | Azure OpenAI, Azure PromptFlow |
| [Create picture generation](./src/06-agents-creative-ag/autogen-gtp4o.ipynb)| Learning about different models to collaborate for better picutres | Azure OpenAI, Autogen |
| [Orchestrating tools](./src/05-tools/app.py)| Learning about orchestrating tools from an application | Azure OpenAI, Azure PromptFlow, LangChain |
| [Orchestrating tools](./src/05-agents-coding-lg/app.py)| Learning about models generating better code | Azure OpenAI, Azure PromptFlow, LangChain |
| [Orchestrating multiple tools](./src/05-multi-tools-tag-lc/app.py)| Learning about multiple orchestrating tools from an application | Azure OpenAI, Azure PromptFlow, LangChain |
| [Agentic RAG](./src/06-agents-lg/app.py)| Learning about using agentic retrieval augmented generation | Azure OpenAI, Azure PromptFlow, LangChain, Langgraph |
| [Async Agents Tools](./src/06-agents-tools-li/app.py)| Learning about using agentic async tools | Azure OpenAI, Azure PromptFlow, Lama Index |
| [Distributed Async Agents Tools](./src/06-agents-kafka-li/app.py)| Learning about using agentic tools in a distributed system | Azure OpenAI, Azure PromptFlow, Lama Index, Kafka |

## Quickstart & Infrastructure setup

The following lines of code will connect your Codespace az cli and azd cli to the right Azure subscription:

```
az login

azd auth login

```

Now deploy the infrastructure components with azure cli

```
azd up
```

Get the values for some env variables
```
azd env get-values | grep AZURE_ENV_NAME
source <(azd env get-values)
```

Last but not least: deploy a dummy container in Azure Container Apps. 
```
bash ./azd-hooks/deploy.sh 03-rag $AZURE_ENV_NAME

```

## Start locally

```
python -m streamlit run app.py --server.port=8000
```

## Deploy resources for Chapter 03

Run the following script

```
azd env get-values | grep AZURE_ENV_NAME
source <(azd env get-values | grep AZURE_ENV_NAME)
bash ./azd-hooks/deploy.sh 06-agents-reports-lg $AZURE_ENV_NAME
```

All the other chapters work the same.

### Configure prompt flow tracing

https://microsoft.github.io/promptflow/reference/pf-command-reference.html
