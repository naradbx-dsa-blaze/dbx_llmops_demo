# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.18.0 mlflow==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import urllib
import json
import mlflow
import MlflowClient, EndpointApiClient

catalog = 'guanyu_chen'
db = 'dbml'

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
model_name = f"{catalog}.{db}.db_chatbot_model"
serving_endpoint_name = f"db_endpoint_{catalog}_{db}"[:63]
latest_model = client.get_model_version_by_alias(model_name, "prod")

#TODO: use the sdk once model serving is available.
serving_client = EndpointApiClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": db,
    "table_name_prefix": serving_endpoint_name
    }
environment_vars={"DATABRICKS_TOKEN": "{{secrets/dbrag/rag_sp_token}}"}
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, model_name=model_name, model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True, auto_capture_config=auto_capture_config, environment_vars=environment_vars)

# COMMAND ----------

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

serving_client.query_inference_endpoint(
    serving_endpoint_name,
    {
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"},
            {
                "role": "assistant",
                "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics.",
            },
            {"role": "user", "content": "Does it support streaming?"},
        ]
    },
)
