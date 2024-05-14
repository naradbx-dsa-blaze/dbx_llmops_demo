# Databricks notebook source
import requests
import json

# Set the name of the MLflow endpoint
endpoint_name = "ft_medbrief_7b"

# Name of the registered MLflow model
model_name = "ang_nara_catalog.llmops.medbrief-7b"

# Get the latest version of the MLflow model
model_version = 2

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

optimizable_info = requests.get(
    url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}",
    headers=headers
).json()

if 'optimizable' not in optimizable_info or not optimizable_info['optimizable']:
    raise ValueError("Model is not eligible for provisioned throughput")

chunk_size = optimizable_info['throughput_chunk_size']

# Minimum desired provisioned throughput
min_provisioned_throughput = 2 * chunk_size

# Maximum desired provisioned throughput
max_provisioned_throughput = 3 * chunk_size

# Send the POST request to create the serving endpoint
data = {
   "name":"ft_mistral7b_endpoint",
   "creator":"narasimha.kamathardi@databricks.com",
   "config":{
      "served_entities":[
         {
            "name":"medbrief-7b-2",
            "entity_name":"ang_nara_catalog.llmops.medbrief-7b",
            "entity_version":"2",
            "workload_size":"Small",
            "workload_type":"GPU_MEDIUM",
            "optimization_config":{
               "llm_optimized":True
            },
            "min_provisioned_throughput": min_provisioned_throughput,
            "max_provisioned_throughput": max_provisioned_throughput
         }
      ],
      "traffic_config":{
         "routes":[
            {
               "served_model_name":"medbrief-7b-2",
               "traffic_percentage":100,
               "served_entity_name":"medbrief-7b-2"
            }
         ]
      },
      "auto_capture_config":{
         "catalog_name":"ang_nara_catalog",
         "schema_name":"llmops",
         "table_name_prefix":"ft_mistral7b_endpoint",

         "enabled":True
      }
   }
}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/serving-endpoints",
    json=data,
    headers=headers
)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import json
import time

client = get_deploy_client("databricks")
endpoint = client.get_endpoint(endpoint="dbdemos_fsi_fraud")

def check_state(endpoint):
    data = json.loads(endpoint)
    state = data.get('state', {}).get('ready', '')

    while state != 'READY':
        # Sleep for a while before checking again
        time.sleep(300)
        # Reload the data to get the updated state
        data = json.loads(endpoint)
        state = data.get('state', {}).get('ready', '')

    return 1
  
# Triggering the function
json_data = json.dumps(endpoint)
result = check_state(json_data)
print(result)

# COMMAND ----------

# Triggering the function
json_data = json.dumps(endpoint)
result = check_state(json_data)
print(result)
