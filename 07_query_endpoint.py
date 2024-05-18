# Databricks notebook source
import gradio as gr
import mlflow.deployments
from databricks.sdk import WorkspaceClient
import re

def filter_incomplete_sentence(text):
    pattern = r'(?:[^.!?]+(?:[.!?](?=\s|$))+\s?)'
    filtered_sentence = ''.join(re.findall(pattern, text))
    return filtered_sentence.strip()

def summarize_medical_history(prompt):
    # Get the deployment client
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    # Define input for the model
    input_data = {"prompt": prompt}
    # Make prediction
    response = deploy_client.predict(endpoint="ft_mistral7b_endpoint", inputs=input_data)
    # Extract and return the response
    get_text = response['choices'][0]['text']
    summary = filter_incomplete_sentence(get_text)
    return summary

# Define the input component
input_text = gr.Textbox(lines=20, label="Enter the detailed notes here", placeholder="Paste clinical notes here...")
# Define the output component
output_text = gr.Textbox(label="Summary", readonly=True)
# Create Gradio interface
gr.Interface(fn=summarize_medical_history, inputs=input_text, outputs=output_text, title="Clinical Notes Summarization").launch()
