# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC %pip install redis gradio mlflow databricks-sdk
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

import gradio as gr
import mlflow.deployments
from databricks.sdk import WorkspaceClient
import re
import sqlite3

class MedicalHistorySummarizer:
    def __init__(self, db_path=':memory:'):
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")
        self.db_path = db_path
        self._setup_db()

    def _setup_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS client_request_id_counter (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            ''')
            cursor.execute('''
                INSERT INTO client_request_id_counter (id) VALUES (0) 
                ON CONFLICT(id) DO NOTHING
            ''')
            conn.commit()

    def get_next_client_request_id(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE client_request_id_counter
                SET id = id + 1
            ''')
            conn.commit()
            cursor.execute('''
                SELECT id FROM client_request_id_counter
            ''')
            result = cursor.fetchone()
            return result[0]

    def filter_incomplete_sentence(self, text):
        pattern = r'(?:[^.!?]+(?:[.!?](?=\s|$))+\s?)'
        filtered_sentence = ''.join(re.findall(pattern, text))
        return filtered_sentence.strip()

    def summarize_medical_history(self, prompt):
        # Increment and get the next client_request_id_counter from SQLite
        client_request_id = self.get_next_client_request_id()
        # Define input for the model
        input_data = {"prompt": prompt, "client_request_id": str(client_request_id)}
        # Make prediction
        response = self.deploy_client.predict(endpoint="ft_mistral7b_endpoint", inputs=input_data)
        # Extract and return the response
        get_text = response['choices'][0]['text']
        summary = self.filter_incomplete_sentence(get_text)
        return summary

# Create an instance of the summarizer
summarizer = MedicalHistorySummarizer(db_path='client_request_id_counter.db')

# Define the input component
input_text = gr.Textbox(lines=20, label="Enter the detailed notes here", placeholder="Paste clinical notes here...")
# Define the output component
output_text = gr.Textbox(label="Summary")

# Create Gradio interface
gr.Interface(fn=summarizer.summarize_medical_history, inputs=input_text, outputs=output_text, title="Clinical Notes Summarization").launch(share=True)

