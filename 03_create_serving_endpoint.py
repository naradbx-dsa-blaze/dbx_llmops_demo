# Databricks notebook source
import requests
import json

# Set the name of the MLflow endpoint
endpoint_name = "ift-medbrief8b-endpoint"

# Name of the registered MLflow model
model_name = "nara_catalog.ds_demos.ift-medbrief8b"

# Get the latest version of the MLflow model
model_version = 2

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"}

# Check if the endpoint already exists
existing_endpoints = requests.get(
    url=f"{API_ROOT}/api/2.0/serving-endpoints",
    headers=headers
).json()

if any(endpoint['name'] == endpoint_name for endpoint in existing_endpoints.get('endpoints', [])):
    print(f"Endpoint '{endpoint_name}' already exists. Skipping creation.")
else:
    optimizable_info = requests.get(
        url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}",
        headers=headers
    ).json()

    if 'optimizable' not in optimizable_info or not optimizable_info['optimizable']:
        raise ValueError("Model is not eligible for provisioned throughput")

    chunk_size = optimizable_info['throughput_chunk_size']

    # Minimum desired provisioned throughput
    min_provisioned_throughput = chunk_size

    # Maximum desired provisioned throughput
    max_provisioned_throughput = 2 * chunk_size

    # Send the POST request to create the serving endpoint
    data = {
       "name": endpoint_name,
       "creator":"narasimha.kamathardi@databricks.com",
       "config":{
          "served_entities":[
             {
                "name":"ift-medbrief8b-2",
                "entity_name": model_name,
                "entity_version": str(model_version),
                "environment_vars":{
                     "ENABLE_MLFLOW_TRACING": True
               },
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
                   "served_model_name":"ift-medbrief8b-2",
                   "traffic_percentage":100,
                   "served_entity_name":"ift-medbrief8b-2"
                }
             ]
          },
          "auto_capture_config":{
             "catalog_name":"ang_nara_catalog",
             "schema_name":"llmops",
             "table_name_prefix": endpoint_name,
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

