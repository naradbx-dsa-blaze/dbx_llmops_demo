# Databricks notebook source
import cloudpickle
import langchain
from mlflow.models import infer_signature

catalog = 'guanyu_chen'
db = 'dbml'

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.db_chatbot_model"

with mlflow.start_run(run_name="db_chatbot_rag") as run:
    #Get our model signature from input/output
    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True,
    )

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)
