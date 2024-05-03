# Databricks notebook source
!pip install -q accelerate==0.27.2 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.38.2 trl==0.4.7 guardrail-ml==0.0.12 mlflow
!pip install -q unstructured["local-inference"]==0.7.4 pillow
!pip install pydantic==1.8.2
dbutils.library.restartPython()
%env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# COMMAND ----------

import mlflow
import transformers
import torch
import mlflow
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

model = AutoModelForCausalLM.from_pretrained("geekdom/medmistral-7b")
tokenizer = AutoTokenizer.from_pretrained("geekdom/medmistral-7b")
registered_model_name = "ang_nara_catalog.llmops.medbrief-7b"
model_config = {
    "max_length": 800,
}

with mlflow.start_run():
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        model_config=model_config,
        artifact_path="model",
        task="llm/v1/completions",
        registered_model_name=registered_model_name
    )
