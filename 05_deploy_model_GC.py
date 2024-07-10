# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.28.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Helper function
from mlflow import MlflowClient

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, AutoCaptureConfigInput, AutoCaptureState,PayloadTable
import json

catalog = 'guanyu_chen'
db = 'dbml'
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

model_name = f"{catalog}.{db}.db_chatbot_model"

table_name_prefix = 'db_rag'

serving_endpoint_name = f"db_endpoint_{catalog}_{db}"[:63]
latest_model_version = get_latest_model_version(model_name)
workload_size = 'Small'
scale_to_zero_enabled = True

w = WorkspaceClient()

endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize(
                workload_size),
            scale_to_zero_enabled=scale_to_zero_enabled,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/dbrag/rag_pat_token}}"
            }
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name=catalog,
        schema_name=db,
        table_name_prefix=table_name_prefix,
        enabled=True
    )
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)

# COMMAND ----------

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

data = {
  "messages": [
    {
      "role": "user",
      "content": "What is Apache Spark?"
    },
    {
      "role": "assistant",
      "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."
    },
    {
      "role": "user",
      "content": "Does it support streaming?"
    }
  ]
}

# COMMAND ----------

import requests
import os, json

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbrag", "rag_pat_token")

API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

url = f"{host}/serving-endpoints/{serving_endpoint_name}/invocations"
# headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
headers = {'Authorization': f'Bearer {API_TOKEN}', 'Content-Type': 'application/json'}
response = requests.request(method='POST', headers=headers, url=url, data=json.dumps(data))
if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
print(response.json())
