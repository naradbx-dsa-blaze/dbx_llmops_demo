# Databricks notebook source
import mlflow.deployments
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

deploy_client = mlflow.deployments.get_deploy_client("databricks")

prompt = "Summarize the patients medical history, including any relevant past illnesses, surgeries, or chronic conditions.\n\n### Discharge Summary Patient Name: Unavailable Gender: Female Age: 32 years Medical Record No.: Unavailable Admission Date: Unavailable Discharge Date: Unavailable Reason for Admission: The 32-year-old female presented with a sudden increase in the size of the solitary nodule in the left lobe of the thyroid with hypothyroidism since 6 months. Hospital Course: The patient underwent a total thyroidectomy as fine needle aspiration cytology (FNAC) of the nodule showed features of HT with papillary carcinoma. During the surgery, nodularity was found on the outer surface. The left lobe was measured 4 cm × 2.5 cm × 1.5cm and the right lobe measured 3 cm × 2 cm × 1 cm. There were areas of HT with prominent lymphoid follicles having germinal center and atrophied thyroid follicles lined by hurthle cells. A tiny focus of follicular variant of papillary carcinoma was also seen. Another focus showed effacement of the thyroid parenchyma by diffuse monotonous lymphoid infiltrate suggestive of NHL (B-cell lineage). Treatment: Levothyroxine was initiated at 300mcg/day and the patient was treated with chemotherapy (R-CHOP regime). Discharge Condition: The patient tolerated the chemotherapy well. At 12 months of follow-up, no recurrence or metastasis was noted. Diagnosis: HT coexisting with papillary carcinoma and primary NHL (B-cell lineage). Follow-Up Care: The patient is advised to undergo regular follow-up care to monitor her health and ensure timely intervention in case of any changes or complications.### Response:"

input = {"prompt": prompt, "max_tokens": 150}

response = deploy_client.predict(endpoint="ft_mistral7b_endpoint", inputs=input)

# Print the response to inspect the structure
print(response)

# COMMAND ----------

# MAGIC %pip install gradio==3.48.0

# COMMAND ----------

dbutils.library.restartPython()
