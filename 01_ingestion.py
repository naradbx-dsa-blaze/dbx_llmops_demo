# Databricks notebook source
import pyspark.pandas as ps
import pyspark.sql.utils
import pandas as pd
import re
import dlt
from pyspark.sql.functions import *
ps.set_option('compute.ops_on_diff_frames', True)

# COMMAND ----------

@dlt.table
def load_data():
  data = pd.read_csv(
        "/Volumes/ang_nara_catalog/llmops/data/synthetic.csv"
    )
  return spark.createDataFrame(data)


# COMMAND ----------

@dlt.table
def clean_data():
  df = dlt.read('load_data')
  df = df.filter(df.task == 'Summarization')
  df = df.select('patient_id', 'note', 'answer')
  df = df.withColumnRenamed("answer", "summary")
  return df

# COMMAND ----------

# Define the function to format the text
def format_data_udf(note):
    return f"Summarize the patients medical history, including any relevant past illnesses, surgeries, or chronic conditions.\n###Instruction:{note}\n\n###Response:\n"
# Register the function as a UDF
format_data_udf = udf(format_data_udf, StringType())

# COMMAND ----------

@dlt.table
def format_notes():
  df = dlt.read('clean_data')
  df = df.withColumn("prompt", format_data_udf(df["note"]))
  df = df.withColumnRenamed("summary", "response")
  return df

# COMMAND ----------

@dlt.table
def create_train_data():
  df = dlt.read('format_notes')
  df = df.limit(2000)
  output_path = "/Volumes/ang_nara_catalog/llmops/data/train.jsonl"
  df.write.json(output_path, mode="overwrite", lineSep="\n")
  df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("readChangeFeed", "true").saveAsTable("ang_nara_catalog.llmops.train_clinical_data")  
  return df

# COMMAND ----------

@dlt.table
def create_test_data():
  df = dlt.read('format_notes')
  sorted_df = df.orderBy(col("patient_id").desc())
  last_500_rows = sorted_df.limit(500)
  last_500_rows = last_500_rows.drop("note")
  output_path = "/Volumes/ang_nara_catalog/llmops/data/test.jsonl"
  last_500_rows.write.json(output_path, mode="overwrite", lineSep="\n")
  last_500_rows.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("readChangeFeed", "true").saveAsTable("ang_nara_catalog.llmops.test_clinical_data")    
  return last_500_rows
