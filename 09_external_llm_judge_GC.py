# Databricks notebook source
from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")

try:
    endpoint_name  = "dbdemos-azure-openai"
    deploy_client.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [
                {
                    "name": endpoint_name,
                    "external_model": {
                        "name": "gpt-35-turbo",
                        "provider": "openai",
                        "task": "llm/v1/chat",
                        "openai_config": {
                            "openai_api_type": "azure",
                            "openai_api_key": "{{secrets/dbdemos/azure-openai}}", #Replace with your own azure open ai key
                            "openai_deployment_name": "dbdemo-gpt35",
                            "openai_api_base": "https://dbdemos-open-ai.openai.azure.com/",
                            "openai_api_version": "2023-05-15"
                        }
                    }
                }
            ]
        }
    )
except Exception as e:
    if 'RESOURCE_ALREADY_EXISTS' in str(e):
        print('Endpoint already exists')
    else:
        print(f"Couldn't create the external endpoint with Azure OpenAI: {e}. Will fallback to llama2-70-B as judge. Consider using a stronger model as a judge.")
        endpoint_name = "databricks-llama-2-70b-chat"

#Let's query our external model endpoint
answer_test = deploy_client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is Apache Spark?"}]})
answer_test['choices'][0]['message']['content']

# COMMAND ----------


