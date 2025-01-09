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
        "/Volumes/nara_catalog/ds_demos/raw_notes/synthetic.csv"
    )
  return spark.createDataFrame(data)


# COMMAND ----------

@dlt.table
def clean_data():
  df = dlt.read('load_data')
  df = df.filter(df.task == 'Summarization')
  df = df.select('patient_id', 'note', 'answer', 'question')
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
  df = df.drop('question')
  df = df.withColumn("prompt", format_data_udf(df["note"]))
  df = df.withColumnRenamed("summary", "response")
  return df

# COMMAND ----------

@dlt.table
def create_train_data():
  df = dlt.read('format_notes')
  df = df.limit(15000)
  df=df.drop("note", "patient_id")
  df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("readChangeFeed", "true").saveAsTable("nara_catalog.ds_demos.train_clinical_data")  
  return df

# COMMAND ----------

@dlt.table
def create_test_data():
  df = dlt.read('format_notes')
  sorted_df = df.orderBy(col("patient_id").desc())
  last_5000_rows = sorted_df.limit(5000)
  last_5000_rows = last_5000_rows.drop("note", "patient_id")
  last_5000_rows.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("readChangeFeed", "true").saveAsTable("nara_catalog.ds_demos.test_clinical_data")    
  return last_5000_rows
