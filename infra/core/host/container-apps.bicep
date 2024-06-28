param name string
param location string = resourceGroup().location
param tags object = {}
@description('User assigned identity name')
param identityName string = ''
param openaiName string
param dynamcSessionsName string

param containerAppsEnvironmentName string
param containerRegistryName string
param logAnalyticsWorkspaceName string
param applicationInsightsName string

module containerAppsEnvironment 'container-apps-environment.bicep' = {
  name: '${name}-container-apps-environment'
  params: {
    name: containerAppsEnvironmentName
    location: location
    tags: tags
    logAnalyticsWorkspaceName: logAnalyticsWorkspaceName
    applicationInsightsName: applicationInsightsName
  }
}

module containerRegistry 'container-registry.bicep' = {
  name: '${name}-container-registry'
  params: {
    name: containerRegistryName
    location: location
    tags: tags
  }
}

resource userIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

module openaiAccess '../security/openai-access.bicep' = {
  name: '${deployment().name}-openai-access'
  params: {
    openAiName: openaiName
    principalId: userIdentity.properties.principalId
  }
}

module containerRegistryAccess '../security/registry-access.bicep' = {
  name: '${deployment().name}-registry-access'
  params: {
    containerRegistryName: containerRegistryName
    principalId: userIdentity.properties.principalId
  }
}

module searchAccess '../security/sessions-access.bicep' = {
  name: '${deployment().name}-dynamic-sessions'
  params: {
    dynamicSessionsName: dynamcSessionsName
    principalId: userIdentity.properties.principalId
  }
}

output defaultDomain string = containerAppsEnvironment.outputs.defaultDomain
output environmentName string = containerAppsEnvironment.outputs.name
output registryLoginServer string = containerRegistry.outputs.loginServer
output registryName string = containerRegistry.outputs.name
