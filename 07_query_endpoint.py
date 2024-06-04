# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

import gradio as gr
import mlflow.deployments
from databricks.sdk import WorkspaceClient
import re
import os

class MedicalHistorySummarizer:
    counter_file = 'client_request_id_counter.txt'

    def __init__(self):
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")
        self.load_counter()

    def load_counter(self):
        if os.path.exists(MedicalHistorySummarizer.counter_file):
            with open(MedicalHistorySummarizer.counter_file, 'r') as file:
                MedicalHistorySummarizer.client_request_id_counter = int(file.read())
        else:
            MedicalHistorySummarizer.client_request_id_counter = 0

    def save_counter(self):
        with open(MedicalHistorySummarizer.counter_file, 'w') as file:
            file.write(str(MedicalHistorySummarizer.client_request_id_counter))

    def filter_incomplete_sentence(self, text):
        pattern = r'(?:[^.!?]+(?:[.!?](?=\s|$))+\s?)'
        filtered_sentence = ''.join(re.findall(pattern, text))
        return filtered_sentence.strip()

    def summarize_medical_history(self, prompt):
        # Increment the class's client_request_id_counter
        MedicalHistorySummarizer.client_request_id_counter += 1
        # Save the updated counter to the file
        self.save_counter()
        # Define input for the model
        input_data = {"prompt": prompt, "client_request_id": str(MedicalHistorySummarizer.client_request_id_counter)}
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
