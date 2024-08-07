# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.5 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = 'ang_nara_catalog'
db = 'llmops'
volume_folder = f"/Volumes/{catalog}/{db}/data"

# COMMAND ----------

# DBTITLE 1,Ingesting PDF files as binary format using Databricks cloudFiles (Autoloader)
# Use Databricks Autoloader to incrementally ingeset new files, making it easy to incrementally consume billions of files from the data lake in various data formats.
# Autoloader easily ingests our unstructured PDF data in binary format.
df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobFilter", "*.pdf")
        .load('dbfs:'+volume_folder+"/raw_notes"))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/raw_clinical_docs')
  .table(f'{catalog}.{db}.pdf_raw').awaitTermination())

# COMMAND ----------

#install poppler on the cluster (should be done by init scripts)
def install_ocr_on_nodes():
    """
    install poppler on the cluster (should be done by init scripts)
    """
    # from pyspark.sql import SparkSession
    import subprocess
    num_workers = max(1,int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")))
    command = "sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y" 
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    # spark = SparkSession.builder.getOrCreate()
    data = spark.sparkContext.parallelize(range(num_workers), num_workers) 
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect();
        print("OCR libraries installed")
    except Exception as e:
        print(f"Couldn't install on all node: {e}")
        raise e

# COMMAND ----------

install_ocr_on_nodes()

# COMMAND ----------

from unstructured.partition.auto import partition
import re

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------

import io
import re
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer
from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator
from pyspark.sql import functions as F

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 5)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below BGE 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What medications were administered to the patient during their hospital stay for the treatment of shortness of breath, cough, myalgias, and malaise, as mentioned in the discharge summary?"]})
print(embeddings)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS ang_nara_catalog.llmops.databricks_pdf_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ang_nara_catalog.llmops.pdf_raw limit 10

# COMMAND ----------

(spark.readStream.table(f'{catalog}.{db}.pdf_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .selectExpr('path as url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_chunk')
    .table(f'{catalog}.{db}.databricks_pdf_documentation').awaitTermination())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ang_nara_catalog.llmops.databricks_pdf_documentation limit 10
