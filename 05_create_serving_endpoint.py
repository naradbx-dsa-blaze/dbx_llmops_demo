# Databricks notebook source
import requests
import json

# Set the name of the MLflow endpoint
endpoint_name = "ft_medbrief_7b"

# Name of the registered MLflow model
model_name = "ang_nara_catalog.llmops.medbrief-7b"

# Get the latest version of the MLflow model
model_version = 1

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
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": model_version,
                "min_provisioned_throughput": min_provisioned_throughput,
                "max_provisioned_throughput": max_provisioned_throughput,
            }
        ]
    },
    "auto_capture_config":{
       "catalog_name": "ang_nara_catalog",
       "schema_name": "llmops",
       "table_name_prefix": "medbrief_7b_payload"
    }
}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/serving-endpoints",
    json=data,
    headers=headers
)

print(json.dumps(response.json(), indent=4))
