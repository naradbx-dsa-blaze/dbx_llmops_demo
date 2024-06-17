# Databricks notebook source
df = spark.read.table("ang_nara_catalog.llmops.ft_mistral7b_endpoint_processed_profile_metrics")

# COMMAND ----------

# Wait for a week (7 days)
print("Waiting for a week to calculate the average...")
time.sleep(7 * 24 * 60 * 60)  # 7 days in seconds

#compute average
average = df.agg({"avg_rouge": "avg"}).collect()[0][0]

# COMMAND ----------

dbutils.jobs.taskValues.set(key = "rouge", value = average)

# COMMAND ----------

dbutils.jobs.taskValues.get(taskKey="09_avg_rouge_calc", key="rouge", default=average, debugValue=average)
