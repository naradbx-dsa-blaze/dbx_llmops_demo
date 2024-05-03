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

@dlt.table
def add_instruction():
  df = dlt.read('clean_data')
  df = df.withColumn("instruction", lit('Summarize the patients medical history, including any relevant past illnesses, surgeries, or chronic conditions')) 
  df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("readChangeFeed", "true").saveAsTable("ang_nara_catalog.llmops.refined_clinical_data")    
  return df
