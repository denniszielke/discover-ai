# Discover AI
This repository contains AI demos

Regions that this deployment can be executed:
- uksouth
- swedencentral
- canadaeast
- australiaeast

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
bash ./azd-hooks/deploy.sh 03-rag $AZURE_ENV_NAME
```

All the other chapters work the same.
