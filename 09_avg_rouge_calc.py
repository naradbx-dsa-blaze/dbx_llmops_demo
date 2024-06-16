# Databricks notebook source
df = spark.read.table("ang_nara_catalog.llmops.ft_mistral7b_endpoint_processed_profile_metrics")

# COMMAND ----------

#define baseline rouge
baseline = 0.20

#compute average
average = df.agg({"avg_rouge": "avg"}).collect()[0][0]


# COMMAND ----------

dbutils.jobs.taskValues.set(key = "rouge", value = average)

# COMMAND ----------

dbutils.jobs.taskValues.get(taskKey="09_avg_rouge_calc", key="rouge", default=average, debugValue=average)
