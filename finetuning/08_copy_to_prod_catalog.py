# Databricks notebook source
import mlflow
from mlflow.exceptions import MlflowException

mlflow.set_registry_uri("databricks-uc")

client = mlflow.tracking.MlflowClient()
src_model_name = "ang_nara_catalog.llmops.ift-medbrief8b"
src_model_version = "2"
src_model_uri = f"models:/{src_model_name}/{src_model_version}"
dst_model_name = "nara_catalog.llmops.ift-medbrief8b-prod"

# Check if the destination model version already exists
def model_version_exists(client, model_name, src_version):
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        if version.version == src_version:
            return True
    return False

if not model_version_exists(client, dst_model_name, src_model_version):
    try:
        copied_model_version = client.copy_model_version(src_model_uri, dst_model_name)
        print(f"Model version copied successfully: {copied_model_version.version}")
    except MlflowException as e:
        print(f"Failed to copy model version: {e}")
else:
    print("Model version already exists in the destination. Skipping copy.")

