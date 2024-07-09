# Databricks notebook source
# MAGIC %pip install --upgrade databricks-agents
# MAGIC %pip install transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.agents import deploy
import os

# Define your input data
input_data = [{
  "prompt" : "Summarize the patients medical history, including any relevant past illnesses, surgeries, or chronic conditions ###Instruction:Hospital Course:\
The patient, a 90-year-old male with a history of pulmonary koch’s and a heavy smoking habit, presented with a chief complaint of cough with expectoration and breathlessness for one month. Chest radiographs showed a large lung mass with multiple rounded opacities. Computed tomography (CT) scan of the chest showed a highly invasive, bulky, cavitating, heterogenous mass with lobulated margins measuring approximately 8.7 x 8.2 x 7.4 cm in the lower lobe of right lung. The mass was invading the pleura and chest wall with associated pleural thickening and subtle rib erosion. Numerous metastatic masses were seen in both lung fields, one of which was invading the mediastinum. Multiple small mediastinal lymph nodes were also seen. The primary tumor was diagnosed as carcinosarcoma consisting mainly of squamous cell carcinoma and component of osteosarcoma with foci of metaplastic osteosarcomatous component.Discharge Summary:\
The patient was referred to another hospital for treatment of malignancy. Patient refused further treatment. Based on chest CT, biopsy, and FNAC findings, a diagnosis of carcinosarcoma was made. No further treatment was given."}]

# COMMAND ----------


from databricks.agents import deploy
from mlflow.utils import databricks_utils as du

deployment = deploy("ang_nara_catalog.llmops.ift-medbrief8b", 2)

# query_endpoint is the URL that can be used to make queries to the app
deployment.query_endpoint

# Copy deployment.rag_app_url to browser and start interacting with your RAG application.
deployment.rag_app_url

