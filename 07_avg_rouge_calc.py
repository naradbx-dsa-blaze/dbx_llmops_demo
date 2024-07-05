# Databricks notebook source
import time

# COMMAND ----------

# Function to check if a table exists in Spark
def table_exists(spark, table_name):
    try:
        # Attempt to read the table
        spark.read.table(table_name)
        return True
    except:
        # If an exception occurs, the table does not exist
        return False

# Define the table name
table_name = "ang_nara_catalog.llmops.`ift-medbrief8b-endpoint_processed_profile_metrics`"

# Wait until the table is created
while not table_exists(spark, table_name):
    print(f"Waiting for the table {table_name} to be created...")
    time.sleep(60)  # Check every 60 seconds

print(f"Table {table_name} is now available.")

# Read the table into a DataFrame
df = spark.read.table(table_name)

# Function to compute the ROUGE average
def compute_rouge_average(df):
    # Aggregate the 'avg_rouge' column to calculate the average
    average = df.agg({"avg_rouge": "avg"}).collect()[0][0]
    return average

# Compute the initial ROUGE average
rouge_average = compute_rouge_average(df)
print(f"Initial ROUGE Average: {rouge_average}")

# Loop to update the average every 2 days
while True:
    time.sleep(2 * 24 * 60 * 60)  # Wait for 2 days
    df = spark.read.table(table_name)  # Re-read the table to get updated data
    rouge_average = compute_rouge_average(df)  # Recompute the ROUGE average
    print(f"Updated ROUGE Average: {rouge_average}")

# COMMAND ----------

 #Set and get the updated ROUGE average using Databricks Utilities
dbutils.jobs.taskValues.set(key = "rouge", value = rouge_average)
dbutils.jobs.taskValues.get(taskKey="09_avg_rouge_calc", key="rouge", default=rouge_average, debugValue=rouge_average)
