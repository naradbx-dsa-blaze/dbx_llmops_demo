# Databricks notebook source
# MAGIC %pip install rouge

# COMMAND ----------

# from mlflow.deployments import get_deploy_client
# deploy_client = get_deploy_client("databricks")

# try:
#     endpoint_name  = "dbdemos-azure-openai"
#     deploy_client.create_endpoint(
#         name=endpoint_name,
#         config={
#             "served_entities": [
#                 {
#                     "name": endpoint_name,
#                     "external_model": {
#                         "name": "gpt-35-turbo",
#                         "provider": "openai",
#                         "task": "llm/v1/chat",
#                         "openai_config": {
#                             "openai_api_type": "azure",
#                             "openai_api_key": "{{secrets/dbdemos/azure-openai}}", #Replace with your own azure open ai key
#                             "openai_deployment_name": "dbdemo-gpt35",
#                             "openai_api_base": "https://dbdemos-open-ai.openai.azure.com/",
#                             "openai_api_version": "2023-05-15"
#                         }
#                     }
#                 }
#             ]
#         }
#     )
# except Exception as e:
#     if 'RESOURCE_ALREADY_EXISTS' in str(e):
#         print('Endpoint already exists')
#     else:
#         print(f"Couldn't create the external endpoint with Azure OpenAI: {e}. Will fallback to llama2-70-B as judge. Consider using a stronger model as a judge.")
#         endpoint_name = "databricks-llama-2-70b-chat"

# #Let's query our external model endpoint
# answer_test = deploy_client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is Apache Spark?"}]})
# answer_test['choices'][0]['message']['content']

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

### load endpoint
import requests
import os, json

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbrag", "rag_pat_token")

API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

catalog = 'guanyu_chen'
db = 'dbml'
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
serving_endpoint_name = f"db_endpoint_{catalog}_{db}"[:63]

url = f"{host}/serving-endpoints/{serving_endpoint_name}/invocations"
# headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
headers = {'Authorization': f'Bearer {API_TOKEN}', 'Content-Type': 'application/json'}
response = requests.request(method='POST', headers=headers, url=url, data=json.dumps(data))
if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
print(response.json())

# COMMAND ----------

# Example testing data
test_data = {
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
    },
    {
      "role": "assistant",
      "content": "Yes, Apache Spark supports real-time data processing through Spark Streaming, which is part of its core API."
    }
  ]
}


# COMMAND ----------

from collections import Counter
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# Semantic Similarity Model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Exact Match
def exact_match_score(prediction, truth):
    return 1 if prediction.strip() == truth.strip() else 0

# F1 Score
def f1_score(prediction, truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Semantic Similarity
def semantic_similarity(prediction, truth):
    pred_embedding = similarity_model.encode(prediction, convert_to_tensor=True)
    truth_embedding = similarity_model.encode(truth, convert_to_tensor=True)
    return util.pytorch_cos_sim(pred_embedding, truth_embedding).item()

# ROUGE Score
def rouge_score(prediction, truth):
    rouge = Rouge()
    scores = rouge.get_scores(prediction, truth)
    return scores[0]['rouge-1']['f']

# Jaccard Similarity
def jaccard_similarity(prediction, truth):
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(truth.lower().split())
    intersection = pred_tokens.intersection(truth_tokens)
    union = pred_tokens.union(truth_tokens)
    return len(intersection) / len(union)

# Log-Likelihood Score (Dummy function as it requires probability outputs)
def log_likelihood(probabilities, truth_index):
    return np.log(probabilities[truth_index])

# COMMAND ----------

def extract_qa_pairs(data):
    messages = data["messages"]
    qa_pairs = []
    for i in range(0, len(messages) - 1, 2):  # Iterate by step of 2 to handle pairs correctly
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            qa_pairs.append((messages[i]["content"], messages[i + 1]["content"]))
    return qa_pairs

# Extract question-answer pairs from the data
test_pairs = extract_qa_pairs(test_data)

results = []
for query, truth in test_pairs:
    # Prepare the data in the format expected by your API
    data = {"messages": [{"role": "user", "content": query}]}

    # Send the request to the model's API
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")

    # Extract the prediction from the response
    response_data = response.json()  # This should be a list of dictionaries
    if not response_data or not isinstance(response_data, list) or not response_data[0]:
        raise ValueError("The API response is empty or not in the expected format.")
    prediction = response_data[0].get('result', '').strip()  # Get 'result' from the first dictionary

    # Compute all scores
    result = {
        "Query": query,
        "Truth": truth,
        "Prediction": prediction,
        "Exact Match": exact_match_score(prediction, truth),
        "F1 Score": f1_score(prediction, truth),
        "Semantic Similarity": semantic_similarity(prediction, truth),
        "ROUGE Score": rouge_score(prediction, truth),
        "Jaccard Similarity": jaccard_similarity(prediction, truth),
        # "Log Likelihood": log_likelihood(probabilities, index)  # This requires probabilities which might not be provided
    }
    results.append(result)

results_df = pd.DataFrame(results)

# COMMAND ----------

display(results_df)
