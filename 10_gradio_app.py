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
        self.feedback_data = []

    def summarize_medical_history(self, prompt, client_request_id):
        # Define input for the model
        input_data = {"prompt": prompt, "client_request_id": client_request_id}
        # Make prediction
        response = self.deploy_client.predict(endpoint="ift-medbrief8b-endpoint", inputs=input_data)
        # Extract and return the response
        summary = response['choices'][0]['text']
        return summary

    def record_feedback(self, summary, feedback):
        # Record the feedback
        self.feedback_data.append({"summary": summary, "feedback": feedback})
        return f"Feedback recorded: {feedback}"

    def get_feedback_data(self):
        # Return the feedback data as a DataFrame
        return pd.DataFrame(self.feedback_data)

# Create an instance of the summarizer
summarizer = MedicalHistorySummarizer()

# Define the input components
input_text = gr.Textbox(lines=20, label="Enter the detailed notes here", placeholder="Paste clinical notes here...")
client_request_id_input = gr.Textbox(label="Client Request ID", placeholder="Enter client request ID here")

# Define the output component
output_text = gr.Textbox(label="Summary")

# Function to update the summary state and display it
def update_summary(prompt, client_request_id):
    summary = summarizer.summarize_medical_history(prompt, client_request_id)
    return summary

# Create Gradio interface
interface = gr.Interface(
    fn=update_summary,
    inputs=[input_text, client_request_id_input],
    outputs=output_text,
    title="Clinical Notes Summarization"
)

# Add event handlers for the feedback buttons
def thumbs_up(summary):
    return summarizer.record_feedback(summary, "yes")

def thumbs_down(summary):
    return summarizer.record_feedback(summary, "no")

# Launch the Gradio app
interface.launch(share=True)
