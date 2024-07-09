# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import gradio as gr
import mlflow.deployments
import pandas as pd
import os

class MedicalHistorySummarizer:
    def __init__(self):
        # Initialize the MLflow deployment client
        try:
            self.deploy_client = mlflow.deployments.get_deploy_client("databricks")
        except Exception as e:
            # Raise an error if the client fails to initialize
            raise RuntimeError(f"Failed to initialize deploy client: {e}")

    def summarize_medical_history(self, prompt, client_request_id):
        try:
            # Prepare input data for the prediction request
            input_data = {"prompt": prompt, "client_request_id": client_request_id}
            # Get the prediction from the deployed model
            response = self.deploy_client.predict(endpoint="ift-medbrief8b-endpoint", inputs=input_data)
            # Extract the summary from the response
            summary = response['choices'][0]['text']
            return summary
        except Exception as e:
            # Return an error message if the prediction fails
            return f"Error during summarization: {e}"

    def save_feedback(self, client_request_id, summary, feedback):
        file_path = "/Volumes/ang_nara_catalog/llmops/data/feedback_data.csv"
        try:
            # Load existing data if file exists, otherwise create an empty DataFrame
            if os.path.isfile(file_path):
                existing_data = pd.read_csv(file_path)
            else:
                existing_data = pd.DataFrame(columns=["client_request_id", "response", "feedback"])

            # Create a new DataFrame for the new feedback
            new_feedback = pd.DataFrame({
                "client_request_id": [client_request_id],
                "response": [summary],
                "feedback": [feedback]
            })

            # Append the new feedback to the existing data
            updated_data = pd.concat([existing_data, new_feedback], ignore_index=True)

            # Write the entire DataFrame back to the CSV file
            updated_data.to_csv(file_path, index=False)

            return f"Feedback recorded: {feedback}"
        except Exception as e:
            return f"Error saving feedback: {e}"

# Instantiate the summarizer
summarizer = MedicalHistorySummarizer()

# Define Gradio UI elements
input_text = gr.Textbox(lines=20, label="Enter the detailed notes here", placeholder="Paste clinical notes here...")
client_request_id_input = gr.Textbox(label="Client Request ID", placeholder="Enter client request ID here")
output_text = gr.Textbox(label="Summary", interactive=False)
feedback_text = gr.Textbox(label="Feedback Status", interactive=False)
submit_button = gr.Button("Submit")
thumbs_up_button = gr.Button("👍")
thumbs_down_button = gr.Button("👎")

# Define the function to handle summarization and feedback preparation
def summarize_and_prepare_feedback(prompt, client_request_id):
    try:
        # Get the summary from the summarizer
        summary = summarizer.summarize_medical_history(prompt, client_request_id)
        return summary, client_request_id
    except Exception as e:
        # Return an error message if summarization fails
        return f"Error during summarization: {e}", client_request_id

# Define the function to handle feedback recording
def record_feedback(summary, client_request_id, feedback):
    try:
        # Save the feedback using the summarizer
        feedback_result = summarizer.save_feedback(client_request_id, summary, feedback)
        return feedback_result
    except Exception as e:
        # Return an error message if saving feedback fails
        return f"Error saving feedback: {e}"

# Define the title of the Gradio interface
title = "Clinical Text Summarization LLM Application"

# Build the Gradio interface
with gr.Blocks(title=title) as interface:
    with gr.Column():
        input_text.render()
        client_request_id_input.render()
        submit_button.render()

    summary_output = gr.Textbox(label="Summary", interactive=False)
    with gr.Row():
        thumbs_up_button.render()
        thumbs_down_button.render()

    feedback_text.render()

    # Set up the submit button to trigger the summarization
    submit_button.click(
        fn=summarize_and_prepare_feedback,
        inputs=[input_text, client_request_id_input],
        outputs=[summary_output, client_request_id_input]
    )

    # Set up the thumbs up button to record positive feedback
    thumbs_up_button.click(
        fn=record_feedback,
        inputs=[summary_output, client_request_id_input, gr.State("Looks quite right!")],
        outputs=[feedback_text]
    )

    # Set up the thumbs down button to record negative feedback
    thumbs_down_button.click(
        fn=record_feedback,
        inputs=[summary_output, client_request_id_input, gr.State("Doesn't look quite right!")],
        outputs=[feedback_text]
    )

# Launch the Gradio interface
interface.launch(share=True)

# COMMAND ----------

import os
file_path = "/Volumes/ang_nara_catalog/llmops/data/feedback_data.csv"
if os.path.isfile(file_path):
  print("yes")
else:
  print("no")
