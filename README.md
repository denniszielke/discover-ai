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
# log in with the provided credentials - OPEN A PRIVATE BROWSER SESSION
az login --use-device-code

# "log into azure dev cli - only once" - OPEN A PRIVATE BROWSER SESSION
azd auth login --use-device-code

```

Now deploy the infrastructure components

```
# "provisioning all the resources with the azure dev cli"
azd up
```

Get the values for some env variables
```
# "get and set the value for AZURE_ENV_NAME"
azd env get-values | grep AZURE_ENV_NAME
source <(azd env get-values)
```

Last but not least: deploy a dummy container in Azure Container Apps. 
```
echo "building and deploying the agent for phase 1"
bash ./azd-hooks/deploy.sh 03-rag $AZURE_ENV_NAME

```

## Deploy resources for Chapter 03

Run the following script

```
azd env get-values | grep AZURE_ENV_NAME
source <(azd env get-values | grep AZURE_ENV_NAME)
bash ./azd-hooks/deploy.sh 03-rag $AZURE_ENV_NAME
```

All the other chapters work the same.
