# Databricks notebook source
!pip install mlflow

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client
import json
import time

client = get_deploy_client("databricks")
endpoint = client.get_endpoint(endpoint="ift-medbrief8b-endpoint")

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

# COMMAND ----------

dbutils.jobs.taskValues.set(key = "status", value = result)

# COMMAND ----------

dbutils.jobs.taskValues.get(taskKey="04_check_endpoint_status", key="status", default=result, debugValue=result)
