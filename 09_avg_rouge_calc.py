# Databricks notebook source
df = spark.read.table("ang_nara_catalog.llmops.`ift-medbrief8b-endpoint_processed_profile_metrics`")

# COMMAND ----------

import time
# Function to compute the ROUGE average
def compute_rouge_average(df):
    average = df.agg({"avg_rouge": "avg"}).collect()[0][0]
    return average

# Compute the initial ROUGE average
rouge_average = compute_rouge_average(df)
print(f"Initial ROUGE Average: {rouge_average}")

# Loop to update the average every 2 days
while True:
    time.sleep(2 * 24 * 60 * 60)  # Wait for 2 days
    rouge_average = compute_rouge_average(df)
    print(f"Updated ROUGE Average: {rouge_average}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key = "rouge", value = rouge_average)

# COMMAND ----------

dbutils.jobs.taskValues.get(taskKey="09_avg_rouge_calc", key="rouge", default=rouge_average, debugValue=rouge_average)
