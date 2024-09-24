param name string
param location string = resourceGroup().location
param tags object = {}

param identityName string
param postgresServerFqdn string
param databaseName string
param databaseAdmin string
param databasePassword string
param nextAuthSecret string
param salt string
param containerAppsEnvironmentName string

resource userIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
}

resource fuseApp 'Microsoft.App/containerApps@2023-04-01-preview' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${userIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      activeRevisionsMode: 'single'
      ingress: {
        external: true
        targetPort: 3000
        transport: 'auto'
      }
      secrets: [
        {
          name: 'DATABASEPASSWORD'
          value: databasePassword
        }
        {
          name: 'NEXTAUTHSECRET'
          value: nextAuthSecret
        }
        {
          name: 'SALT'
          value: salt
        }
      ]
    }
    template: {
      containers: [
        {
          image: 'ghcr.io/langfuse/langfuse:latest'
          name: 'langfuse'
          env: [
            {
              name: 'DATABASE_HOST'
              value: postgresServerFqdn
            }
            {
              name: 'DATABASE_NAME'
              value: databaseName
            }
            {
              name: 'DATABASE_USERNAME'
              value: databaseAdmin
            }
            {
              name: 'DATABASE_PASSWORD'
              secretRef: 'databasepassword'
            }
            {
              name: 'NEXTAUTH_URL'
              value: 'https://${name}.${containerAppEnv.outputs.defaultDomain}'
            }
            {
              name: 'NEXTAUTH_SECRET'
              secretRef: 'nextauthsecret'
            }
            {
              name: 'SALT'
              secretRef: 'salt'
            }
            {
              name: 'ENCRYPTION_KEY'
              secretRef: '0000000000000000000000000000000000000000000000000000000000000000'
            }            
          ]
          resources: {
            cpu: json('1')
            memory: '2.0Gi'
          }
        }
      ]
    }
  }
}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2022-03-01' existing = {
  name: containerAppsEnvironmentName
}

output uri string = 'https://${fuseApp.properties.configuration.ingress.fqdn}'
