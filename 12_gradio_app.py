# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import gradio as gr
import mlflow.deployments
import re

class MedicalHistorySummarizer:
    def __init__(self):
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def filter_incomplete_sentence(self, text):
        pattern = r'(?:[^.!?]+(?:[.!?](?=\s|$))+\s?)'
        filtered_sentence = ''.join(re.findall(pattern, text))
        return filtered_sentence.strip()

    def summarize_medical_history(self, prompt, client_request_id):
        # Define input for the model
        input_data = {"prompt": prompt, "client_request_id": client_request_id}
        # Make prediction
        response = self.deploy_client.predict(endpoint="ift-medbrief8b-endpoint ", inputs=input_data)
        # Extract and return the response
        get_text = response['choices'][0]['text']
        summary = self.filter_incomplete_sentence(get_text)
        return summary

# Create an instance of the summarizer
summarizer = MedicalHistorySummarizer()

# Define the input components
input_text = gr.Textbox(lines=20, label="Enter the detailed notes here", placeholder="Paste clinical notes here...")
client_request_id_input = gr.Textbox(label="Client Request ID", placeholder="Enter client request ID here")

# Define the output component
output_text = gr.Textbox(label="Summary")

# Create Gradio interface
interface = gr.Interface(
    fn=summarizer.summarize_medical_history,
    inputs=[input_text, client_request_id_input],
    outputs=output_text,
    title="Clinical Notes Summarization"
)

# Launch the Gradio app
interface.launch(share=True)
