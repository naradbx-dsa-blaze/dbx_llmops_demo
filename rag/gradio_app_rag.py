# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC %pip install textblob
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import gradio as gr
import pandas as pd
import json
import requests
import os
from textblob import TextBlob

class ClinicalQABot:
    def __init__(self):
        self.serving_endpoint_name = 'db_endpoint_ang_nara_catalog_llmops'
        self.host = 'https://e2-demo-field-eng.cloud.databricks.com'
        self.url = f"{self.host}/serving-endpoints/{self.serving_endpoint_name}/invocations"
        os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbrag", "rag_pat_token")
        self.api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {'Authorization': f'Bearer {self.api_token}', 'Content-Type': 'application/json'}
        self.sentiment_analyzer = TextBlob

    def handle_medical_queries(self, messages, client_request_id):
        try:
            input_data = {
                "dataframe_records": [{"messages": [{"role": role, "content": content} for role, content in messages]}],
                "params": {
                    "temperature": 0.5,
                    "max_tokens": 100,
                    "stop": ["word1", "word2"],
                    "candidate_count": 1
                }
            }
            response = requests.post(headers=self.headers, url=self.url, data=json.dumps(input_data))
            if response.status_code != 200:
                raise Exception(f'Request failed with status {response.status_code}, {response.text}')
            result = response.json()
            predictions = result.get('predictions', [{}])
            response_text = predictions[0].get('result', 'No response available')
            return response_text
        except Exception as e:
            return f"Error during handling query: {e}"

    def save_feedback(self, client_request_id, response, feedback):
        file_path = "/Volumes/ang_nara_catalog/llmops/data/feedback_data.csv"
        try:
            if os.path.isfile(file_path):
                existing_data = pd.read_csv(file_path)
            else:
                existing_data = pd.DataFrame(columns=["client_request_id", "response", "feedback"])
            new_feedback = pd.DataFrame({
                "client_request_id": [client_request_id],
                "response": [response],
                "feedback": [feedback]
            })
            updated_data = pd.concat([existing_data, new_feedback], ignore_index=True)
            updated_data.to_csv(file_path, index=False)
            return f"Feedback recorded: {feedback}"
        except Exception as e:
            return f"Error saving feedback: {e}"

    def analyze_sentiment(self, message):
        blob = self.sentiment_analyzer(message)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.5:
            return "POSITIVE"
        elif sentiment < -0.5:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

# Instantiate the bot
bot = ClinicalQABot()

def chatbot_interaction(messages, user_message, client_request_id):
    sentiment = bot.analyze_sentiment(user_message)
    
    # Process the user's message
    messages.append(["user", user_message])

    # Check if sentiment is positive
    if sentiment == "POSITIVE":
        # Acknowledge positive feedback
        acknowledgment = "Glad I was helpful! If you have any further questions, let me know."
        messages.append(["assistant", acknowledgment])
        return messages, client_request_id

    # Process normal queries
    response = bot.handle_medical_queries(messages, client_request_id)
    messages.append(["assistant", response])
    return messages, client_request_id

def submit_message(user_message, chatbot_history, client_request_id):
    if not user_message:
        return chatbot_history, client_request_id, user_message

    updated_chatbot, updated_client_request_id = chatbot_interaction(chatbot_history, user_message, client_request_id)
    return updated_chatbot, updated_client_request_id, ""  # Return empty string to clear the input box

def handle_feedback(feedback, client_request_id, chatbot_history):
    response = next((text for role, text in reversed(chatbot_history) if role == 'assistant'), 'No response available')
    try:
        feedback_result = bot.save_feedback(client_request_id, response, feedback)
        return feedback_result
    except Exception as e:
        return f"Error saving feedback: {e}"

def clear_chat():
    return [], ""

# Define the title of the Gradio interface
title = "Clinical Q&A Bot"

# Build the Gradio interface
with gr.Blocks(title=title) as interface:
    with gr.Column():
        gr.Markdown("### Welcome to the Clinical Q&A Bot")

        # Chatbot component to display the conversation
        chatbot = gr.Chatbot(label="Chat", height=500)

        # Typing input field and buttons
        with gr.Row():
            user_input = gr.Textbox(
                show_label=False,
                placeholder="Type your message here...",
                lines=1,
                container=False,
                elem_id="chat-input"
            )
            with gr.Column():
                send_button = gr.Button("Send", size="small", elem_id="send-btn")
                clear_chat_button = gr.Button("Clear Chat", size="small", elem_id="clear-chat-btn")

        # Feedback section with header
        gr.Markdown("### We value your feedback! Please let us know how we did:")
        with gr.Row():
            thumbs_up_button = gr.Button("👍", size="small", elem_id="thumbs-up-btn")
            thumbs_down_button = gr.Button("👎", size="small", elem_id="thumbs-down-btn")

        feedback_text = gr.Textbox(label="Feedback Status", interactive=False, lines=2, container=False)

        # Set up interactions
        send_button.click(
            fn=submit_message,
            inputs=[user_input, chatbot, gr.State("client_request_id_placeholder")],
            outputs=[chatbot, gr.State("client_request_id_placeholder"), user_input]
        )

        thumbs_up_button.click(
            fn=handle_feedback,
            inputs=[gr.State("feedback_type", "POSITIVE"), gr.State("client_request_id_placeholder"), chatbot],
            outputs=[feedback_text]
        )

        thumbs_down_button.click(
            fn=handle_feedback,
            inputs=[gr.State("feedback_type", "NEGATIVE"), gr.State("client_request_id_placeholder"), chatbot],
            outputs=[feedback_text]
        )

        clear_chat_button.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, user_input]
        )

# Add custom CSS to style buttons and layout
interface.css = """
#chat-input {
    position: sticky;
    bottom: 0;
    width: calc(100% - 160px); /* Adjusted for both buttons */
    padding: 10px;
    margin: 0;
    box-sizing: border-box;
    background: #ffffff; /* White background */
    border-top: 1px solid #ddd;
}
#send-btn, #thumbs-up-btn, #thumbs-down-btn, #clear-chat-btn {
    font-size: 0.9em;
    border: none;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 5px;
    margin: 0 5px;
    text-align: center;
    white-space: nowrap;
}
#send-btn {
    background: #007bff; /* Soft blue */
    color: white;
}
#clear-chat-btn {
    background: #6c757d; /* Soft gray */
    color: white;
}
#thumbs-up-btn {
    background: #e2e2e2; /* Light gray */
    color: #28a745; /* Soft green */
}
#thumbs-down-btn {
    background: #e2e2e2; /* Light gray */
    color: #dc3545; /* Soft red */
}
#send-btn, #clear-chat-btn {
    max-width: 120px;
}
#thumbs-up-btn, #thumbs-down-btn {
    max-width: 100px;
}
#feedback-section {
    margin-top: 20px;
    text-align: center;
}
"""

# Launch the Gradio interface
interface.launch(share=True)
