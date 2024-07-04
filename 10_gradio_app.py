# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import gradio as gr
import mlflow.deployments
from delta import DeltaTable
import pandas as pd

class MedicalHistorySummarizer:
    def __init__(self):
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")
        
    def summarize_medical_history(self, prompt, client_request_id):
        input_data = {"prompt": prompt, "client_request_id": client_request_id}
        response = self.deploy_client.predict(endpoint="ift-medbrief8b-endpoint", inputs=input_data)
        summary = response['choices'][0]['text']
        return summary

    def save_feedback(self, client_request_id, summary, feedback):
        feedback_data = pd.DataFrame({
            "Client Request ID": [client_request_id],
            "Summary": [summary],
            "Feedback": [feedback]
        })
        feedback_data.write.format("delta").mode("overwrite").saveAsTable("ang_nara_catalog.llmops.user_feedback")

summarizer = MedicalHistorySummarizer()

input_text = gr.Textbox(lines=20, label="Enter the detailed notes here", placeholder="Paste clinical notes here...")
client_request_id_input = gr.Textbox(label="Client Request ID", placeholder="Enter client request ID here")
output_text = gr.Textbox(label="Summary", interactive=False)
feedback_text = gr.Textbox(label="Feedback Status", interactive=False)
submit_button = gr.Button("Submit")
thumbs_up_button = gr.Button("👍")
thumbs_down_button = gr.Button("👎")

def summarize_and_prepare_feedback(prompt, client_request_id):
    summary = summarizer.summarize_medical_history(prompt, client_request_id)
    return summary, client_request_id

def record_feedback(summary, client_request_id, feedback):
    feedback_result = summarizer.save_feedback(client_request_id, summary, feedback)
    return feedback_result

title = "Clinical Text Summarization LLM Application"

with gr.Blocks(title=title) as interface:
    with gr.Column():
        input_text.render()
        client_request_id_input.render()  # Render client_request_id_input before submit button
        submit_button.render()

    summary_output = gr.Textbox(label="Summary", interactive=False)
    with gr.Row():
        thumbs_up_button.render()
        thumbs_down_button.render()

    feedback_text.render()

    submit_button.click(
        fn=summarize_and_prepare_feedback,
        inputs=[input_text, client_request_id_input],
        outputs=[summary_output, client_request_id_input]
    )

    thumbs_up_button.click(
        fn=record_feedback,
        inputs=[summary_output, client_request_id_input, gr.State("Looks quite right!")],
        outputs=[feedback_text]
    )
    thumbs_down_button.click(
        fn=record_feedback,
        inputs=[summary_output, client_request_id_input, gr.State("Dosen't look quite right!")],
        outputs=[feedback_text]
    )

interface.launch(share=True)
