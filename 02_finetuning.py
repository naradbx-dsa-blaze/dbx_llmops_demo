# Databricks notebook source
# # Let's start by installing our products
# %pip install databricks-genai==1.0.2
# %pip install databricks-sdk==0.27.1
# %pip install "mlflow==2.12.2"
# dbutils.library.restartPython()

# COMMAND ----------

# from databricks.model_training import foundation_model as fm
# #Return the current cluster id to use to read the dataset and send it to the fine tuning cluster. See https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html#cluster-id
# def get_current_cluster_id():
#   import json
#   return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']


# #Let's clean the model name
# registered_model_name = "ang_nara_catalog.llmops.ift-medbrief8b"

# run = fm.create(
#     data_prep_cluster_id=get_current_cluster_id(),  # required if you are using delta tables as training data source. This is the cluster id that we want to use for our data prep job.
#     model="meta-llama/Meta-Llama-3-8B-Instruct",  # Here we define what model we used as our baseline
#     train_data_path="ang_nara_catalog.llmops.create_train_data",
#     eval_data_path="ang_nara_catalog.llmops.create_test_data",
#     task_type="INSTRUCTION_FINETUNE",  # Change task_type="INSTRUCTION_FINETUNE" if you are using the fine-tuning API for completion.
#     register_to=registered_model_name,
#     training_duration="15ep", #only 15 epoch to accelerate the demo. Check the mlflow experiment metrics to see if you should increase this number
#     learning_rate="5e-7",
# )

# print(run)

# COMMAND ----------

# display(run.get_events())
