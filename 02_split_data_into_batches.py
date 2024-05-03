# Databricks notebook source
import pandas as pd
import json

# Load the JSON file into a pandas DataFrame
with open("/Volumes/ang_nara_catalog/llmops/data/clinical_notes.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)

# Define the batch size
batch_size = 2000

# Calculate the number of batches
num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

# Split the DataFrame into batches and write each batch to a separate JSON file
for i in range(num_batches):
    start_index = i * batch_size
    end_index = (i + 1) * batch_size
    batch_df = df[start_index:end_index]
    batch_df.to_json(f"/Volumes/ang_nara_catalog/llmops/data/clinical_data_batch_{i}.json", orient="records")
