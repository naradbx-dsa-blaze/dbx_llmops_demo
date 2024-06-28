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
    return f"Summarize the patients medical history, including any relevant past illnesses, surgeries, or chronic conditions.\n\n###Instruction: {note}\n\n###Response:"
# Register the function as a UDF
format_data_udf = udf(format_data_udf, StringType())

# COMMAND ----------

@dlt.table
def format_notes(note):
  df = dlt.read('load_data')
  df = df.withColumn("prompt", format_data_udf(df["note"]))
  df = df.withColumnRenamed("summary", "response")
  train_df = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("readChangeFeed", "true").saveAsTable("ang_nara_catalog.llmops.train_clinical_data") 
  return train_df

# COMMAND ----------

@dlt.table
def create_test_data():
  df = dlt.read('add_instruction')
  sorted_df = df.orderBy(col("patient_id").desc())
  last_10000_rows = sorted_df.limit(10000)
  last_10000_rows.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("readChangeFeed", "true").saveAsTable("ang_nara_catalog.llmops.test_clinical_data")    
  return last_10000_rows
