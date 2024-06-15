# Databricks notebook source
# DBTITLE 1,Load the required libraries
# MAGIC %pip install textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 transformers==4.30.2 rouge torch==1.13.1 "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.14-py3-none-any.whl"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Set endpoint name
endpoint_name = "ft_mistral7b_endpoint"

# COMMAND ----------

import requests
from typing import Dict
import time


def get_endpoint_status(endpoint_name: str) -> Dict:
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response

response = get_endpoint_status(endpoint_name=endpoint_name)

auto_capture_config = response["config"]["auto_capture_config"]
catalog = auto_capture_config["catalog_name"]
schema = auto_capture_config["schema_name"]
# These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
payload_table_name = f"`{catalog}`.`{schema}`.`{payload_table_name}`"
print(f"Endpoint {endpoint_name} configured to log payload in table {payload_table_name}")

processed_table_name = f"{auto_capture_config['table_name_prefix']}_processed"
processed_table_name = f"`{catalog}`.`{schema}`.`{processed_table_name}`"
print(f"Processed requests with text evaluation metrics will be saved to: {processed_table_name}")

payloads = spark.table(payload_table_name)

while payloads.count() < 1:
    print("Waiting for more payloads to be logged...")
    time.sleep(30)  # Adjust the sleep duration as needed
    payloads = spark.table(payload_table_name)

print("Payload count greater than 1. Continuing...")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import from_json, col
from pyspark.sql import DataFrame, functions as F
import json

def convert_string_columns_to_json(df, string_columns):
    """
    Convert string columns to JSON types by dynamically inferring the schema.

    Args:
    - df: PySpark DataFrame
    - string_columns: List of names of string columns to convert to JSON

    Returns:
    - df with the specified string columns converted to JSON
    """
    
    # Define a function to infer the schema dynamically
    def infer_schema(json_str):
        if json_str:
            try:
                json_dict = json.loads(json_str)
                fields = []
                for key, value in json_dict.items():
                    if isinstance(value, dict):
                        sub_fields = [StructField(sub_key, StringType(), True) for sub_key in value.keys()]
                        fields.append(StructField(key, StructType(sub_fields), True))
                    elif isinstance(value, list):
                        if value:
                            sub_fields = [StructField(sub_key, StringType(), True) for sub_key in value[0].keys()]
                            fields.append(StructField(key, ArrayType(StructType(sub_fields)), True))
                        else:
                            fields.append(StructField(key, ArrayType(StringType()), True))
                    else:
                        fields.append(StructField(key, StringType(), True))
                return StructType(fields)
            except ValueError:
                # Return an empty schema if JSON string cannot be parsed
                return StructType([])
        else:
            # Return an empty schema if JSON string is empty or null
            return StructType([])

    # Convert each specified string column to JSON
    for string_column in string_columns:
        schema = infer_schema(df.select(string_column).take(1)[0][0])
        df = df.withColumn(string_column, from_json(col(string_column), schema))

    return df

# COMMAND ----------

string_columns = [
  "request", "response"]
payloads = convert_string_columns_to_json(payloads, string_columns)
# Filter out the non-successful requests.
payloads = payloads.filter(F.col("status_code") == "200")

# COMMAND ----------

#flatten json payload
payloads = payloads.withColumn("request", col("request").prompt)
payloads = payloads.withColumn("response", col("response").choices[0].text)

# COMMAND ----------

#cleanup junk output off of inference table
import re
def filter_incomplete_sentence(text):
    pattern = r"(?:[^.!?]+(?:[.!?](?=\s|$))+\s?)"
    filtered_sentence = "".join(re.findall(pattern, text))
    return filtered_sentence.strip()
  
filter_incomplete_sentence = udf(filter_incomplete_sentence)
  
payloads = payloads.withColumn("response", filter_incomplete_sentence(payloads["response"]))

# COMMAND ----------

from pyspark.sql.functions import row_number, lit
from pyspark.sql.window import Window

#add client_request_id column to the test table
test_df = spark.read.table("ang_nara_catalog.llmops.create_test_data")
test_df = test_df.withColumnRenamed("summary", "ground_truth")
windowSpec = Window.orderBy(lit(1))
test_df = test_df.withColumn("client_request_id", F.row_number().over(windowSpec))

# COMMAND ----------

#join ground_truth table with inference table
joined_df = payloads.join(test_df, "client_request_id", "inner").drop("id")
payloads = joined_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute the Input / Output text evaluation metrics (e.g., toxicity, perplexity, readability) 
# MAGIC
# MAGIC Now that our input and output are unpacked and available as a string, we can compute their metrics. These will be analyzed by Lakehouse Monitoring so that we can understand how these metrics change over time.
# MAGIC
# MAGIC Feel free to add your own custom evaluation metrics here.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType
from rouge import Rouge
import pandas as pd

# Initialize ROUGE scorer
rouge = Rouge()

# Define the Pandas UDF to compute and round up ROUGE score
@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def compute_rouge_score(output_series, ground_truth_series):
    rouge_scores = []
    for output, ground_truth in zip(output_series, ground_truth_series):
        scores = rouge.get_scores(output, ground_truth)
        # Assuming you want to use ROUGE-L, which is scores[0]['rouge-l']['f']
        rouge_l_f_score = scores[0]['rouge-l']['f']
        # Round up the ROUGE score to 2 decimal places
        rounded_rouge_score = round(rouge_l_f_score, 2)
        rouge_scores.append(rounded_rouge_score)
    return pd.Series(rouge_scores)

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
def compute_metrics(requests_df: DataFrame, column_to_measure = ["request", "response"]) -> DataFrame:
  for column_name in column_to_measure:
    requests_df = requests_df.withColumn("rouge_score", compute_rouge_score(requests_df["response"], requests_df["ground_truth"]))
  return requests_df

# COMMAND ----------

# MAGIC %md
# MAGIC We can now incrementally consume new payload from the inference table, unpack them, compute metrics and save them to our final processed table:

# COMMAND ----------

payloads = payloads.drop("date", "status_code", "sampling_fraction", "client_request_id", "databricks_request_id", "note", "request_metadata", "instruction")

# COMMAND ----------

#write processed payloads to delta table
payloads.write.format("delta").mode("overwrite").saveAsTable("ang_nara_catalog.llmops.processed_payloads")

# COMMAND ----------

# Initialize the processed requests table. Turn on CDF (for monitoring) and enable special characters in column names. 
def create_processed_table_if_not_exists(table_name, requests_with_metrics):
    DeltaTable.createIfNotExists(spark) \
        .tableName(table_name) \
        .addColumns(requests_with_metrics.schema) \
        .property("delta.enableChangeDataFeed", "true") \
        .property("delta.columnMapping.mode", "name") \
        .property("delta.minReaderVersion", "2") \
        .property("delta.minWriterVersion", "5") \
        .execute()

# COMMAND ----------

from delta.tables import DeltaTable
import shutil

#define checkpoint location for streaming
checkpoint_location = "/Volumes/ang_nara_catalog/llmops/checkpoint"

#Check whether the table exists before proceeding.
DeltaTable.isDeltaTable(spark, "ang_nara_catalog.llmops.processed_payloads")

#read processed payloads table
requests_raw = spark.readStream.table("ang_nara_catalog.llmops.processed_payloads")

#Compute text evaluation metrics.
requests_with_metrics = compute_metrics(requests_raw)

#Persist the requests stream, with a defined checkpoint path for this table.
create_processed_table_if_not_exists(processed_table_name, requests_with_metrics)

#Delete the existing checkpoint directory
dbutils.fs.rm(checkpoint_location, True)
print(f"Deleted old checkpoint location: {checkpoint_location}")

#Create a new checkpoint location as a volume
dbutils.fs.mkdirs(checkpoint_location)
print(f"Created new checkpoint location: {checkpoint_location}")

#Write the streaming DataFrame to Delta table using foreachBatch
requests_with_metrics.writeStream \
    .trigger(processingTime="10 seconds") \
    .foreachBatch(lambda batch_df, batch_id: batch_df.write.format("delta").mode("append").saveAsTable(processed_table_name)) \
    .option("checkpointLocation", checkpoint_location) \
    .start() \
    .awaitTermination(100)

# COMMAND ----------

from pyspark.sql import types as T
"""
Optional parameters to control monitoring analysis. For help, use the command help(lm.create_monitor).
"""
GRANULARITIES = ["1 day"]                        # Window sizes to analyze data over
SLICING_EXPRS = None                             # Expressions to slice data with
BASELINE_TABLE = None                            # Baseline table name, if any, for computing baseline drift

# COMMAND ----------

from pyspark.sql.types import DoubleType

import databricks.lakehouse_monitoring as lm

monitor_params = {
    "profile_type": lm.TimeSeries(
        timestamp_col="timestamp_ms",
        granularities=GRANULARITIES,
    ),
    "output_schema_name": f"{catalog}.{schema}",
    "schedule": None,  # We will refresh the metrics on-demand in this notebook
    "baseline_table_name": BASELINE_TABLE,
    "slicing_exprs": SLICING_EXPRS,
    "custom_metrics": [
        lm.Metric(
            name="avg_rouge",
            definition="avg(`{{input_column}}`)",
            type="aggregate",
            input_columns=["rouge_score"],
            output_data_type=DoubleType()
        )
    ]
}

try:
    info = lm.create_monitor(table_name=processed_table_name, **monitor_params)
    print(info)
except Exception as e:
    # Ensure the exception was expected
    assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"

    # Update the monitor if any parameters of this notebook have changed.
    lm.update_monitor(table_name=processed_table_name, updated_params=monitor_params)
    # Refresh metrics calculated on the requests table.
    refresh_info = lm.run_refresh(table_name=processed_table_name)
    print(refresh_info)
