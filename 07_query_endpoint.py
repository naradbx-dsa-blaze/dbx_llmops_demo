# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

import gradio as gr
import mlflow.deployments
from databricks.sdk import WorkspaceClient
import re

class MedicalHistorySummarizer:
    def __init__(self):
        self.client_request_id_counter = 0
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def filter_incomplete_sentence(self, text):
        pattern = r'(?:[^.!?]+(?:[.!?](?=\s|$))+\s?)'
        filtered_sentence = ''.join(re.findall(pattern, text))
        return filtered_sentence.strip()

    def summarize_medical_history(self, prompt):
        # Increment the client_request_id_counter
        self.client_request_id_counter += 1
        # Define input for the model
        input_data = {"prompt": prompt, "client_request_id": str(self.client_request_id_counter)}
        # Make prediction
        response = self.deploy_client.predict(endpoint="ft_mistral7b_endpoint", inputs=input_data)
        # Extract and return the response
        get_text = response['choices'][0]['text']
        summary = self.filter_incomplete_sentence(get_text)
        return summary

# Create an instance of the summarizer
summarizer = MedicalHistorySummarizer()

# Define the input component
input_text = gr.Textbox(lines=20, label="Enter the detailed notes here", placeholder="Paste clinical notes here...")
# Define the output component
output_text = gr.Textbox(label="Summary")

# Create Gradio interface
gr.Interface(fn=summarizer.summarize_medical_history, inputs=input_text, outputs=output_text, title="Clinical Notes Summarization").launch(share=True)
