# Databricks notebook source
# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC %pip install --upgrade sqlalchemy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

is_question_about_databricks_str = """
You are classifying documents to know if this question is related to patient clinical history, discharge summary, treatment performed for specific disease  or something from a very different field. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?, classify this question: Do you know the full form of AML?
Expected Response: Yes

Question: Knowing this followup history: What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?, classify this question: Do you know the composition of drug used to cure AML?
Expected Response: Yes

Question: Answer any question related to medicine or patient information: what is the composition of drug Penicillin G?, classify this question: Do you have more details?
Expected Response: Yes

Question: Medical document transaltion to Dutch or French depending upon user question: What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?, classify this question: Translate the answer to French or Dutch depending upon user prompt
Expected Response: Yes

Question: Knowing this followup history: What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_databricks_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_databricks_str
)

is_about_databricks_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?"}, 
        {"role": "assistant", "content": "The patient underwent several treatment regimens for his relapsed AML with concurrent molecular aberration recurrence after remission duration, including venetoclax combined with azacitidine and chidamide with venetoclax and azacitidine regimen. He achieved complete remission after one course of the chidamide combined regimen."}, 
        {"role": "user", "content": "What was this patients discharge summary?"}
    ]
}))

# COMMAND ----------

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "Do you know the composition of drug used to cure AML?"}, 
        {"role": "assistant", "content": "Gemtuzumab ozogamicin combines the monoclonal antibody gemtuzumab, which targets CD33 on leukemia cells, with the cytotoxic agent ozogamicin, an antibiotic that kills the targeted cells. This drug is used to treat relapsed AML by delivering targeted chemotherapy."}, 
        {"role": "user", "content": "What is the recommended dosage for stage III AML?"},
        {"role": "assistant", "content": "For relapsed or refractory AML, Gemtuzumab ozogamicin is typically dosed at 6 mg/m² intravenously over 1 hour on days 1, 8, and 15 of a 28-day cycle. This regimen may vary based on individual patient factors and response. Dosage for stage 3 AML specifically depends on the overall treatment plan and should be tailored by a healthcare provider."}
    ]
}))

# COMMAND ----------

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What was the patient's diagnosis and treatment plan for multiple thrombotic events associated with antithrombin deficiency (ATD) according to the discharge summary?"}, 
        {"role": "assistant", "content": "The patient was diagnosed with deep vein thrombosis (DVT) of the right lower extremity, left lower extremity, both iliac vessels, and inferior vena cava (IVC) associated with antithrombin deficiency (ATD). The treatment plan involved catheter-directed local thrombolysis using recombinant tissue plasminogen activator (rt-PA), followed by anticoagulative therapy with vitamin K antagonist phenprocoumon. To prevent re-thrombosis, oral anticoagulation with phenprocoumon should be continued lifelong and regular follow-up examinations should be done at the outpatient clinic. The patient and their family should be counseled on the importance of continued medical management for ATD. While interventions were done for thrombotic lesions in the right lower extremity and pelvic axis, the left iliac vessels remained thrombotic and required further interventions. Therefore, more interventions should be considered for thrombotic lesions found in the pelvic vessels."}, 
        {"role": "user", "content": "Please translate the above response to French"},
        {"role": "assistant", "content": "Le patient a été diagnostiqué avec une thrombose veineuse profonde (TVP) de l'extrémité inférieure droite, de l'extrémité inférieure gauche, des deux vaisseaux iliaques et de la veine cave inférieure (VCI) associée à une déficience en antithrombine (ATD). Le plan de traitement a impliqué une thrombolyse locale dirigée par cathéter utilisant un activateur recombinant du plasminogène (rt-PA), suivie d'une thérapie anticoagulante avec l'antagoniste de la vitamine K, le phénprocoumone. Pour prévenir la re-thrombose, l'anticoagulation orale avec le phénprocoumone doit être poursuivie à vie et des examens de suivi réguliers doivent être effectués en consultation externe. Le patient et sa famille doivent être informés de l'importance d'une gestion médicale continue pour l'ATD. Bien que des interventions aient été réalisées pour les lésions thrombotiques dans l'extrémité inférieure droite et l'axe pelvien, les vaisseaux iliaques gauches sont restés thrombotiques et nécessitent des interventions supplémentaires. Par conséquent, des interventions supplémentaires doivent être envisagées pour les lésions thrombotiques trouvées dans les vaisseaux pelviens."}
    ]
}))

# COMMAND ----------

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What treatments were given to the 4.3-year-old boy in the given discharge summary for his fibrinoid formation in the right upper lobe, and what was the outcome of those treatments?"}, 
        {"role": "assistant", "content": "The 4.3-year-old boy with fibrinoid formation in the right upper lobe was given cephalothin and azithromycin initially, but due to worsening symptoms and hypoxemia, he required ventilatory support and additional treatment with methylprednisolone. Thoracentesis was performed to remove pleural fluid, and further diagnostic tests confirmed his recovery. He was discharged with oral azithromycin. The outcome of these treatments was a return to normal temperature and recovery from his respiratory infection."}, 
        {"role": "user", "content": "Please translate the above response to Dutch"},
        {"role": "assistant", "content": "De 4,3-jarige jongen met fibrinoïde vorming in de rechter bovenkwab kreeg aanvankelijk cephalotine en azitromycine, maar vanwege verslechterende symptomen en hypoxemie had hij ventilatoire ondersteuning en aanvullende behandeling met methylprednisolon nodig. Een thoracentese werd uitgevoerd om pleuravocht te verwijderen, en verdere diagnostische tests bevestigden zijn herstel. Hij werd ontslagen met orale azitromycine. Het resultaat van deze behandelingen was een terugkeer naar de normale temperatuur en herstel van zijn luchtweginfectie."}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about patient history
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about patient history
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the color of Water?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about patient history
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is capital of India?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about patient history
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the scientific name of Cat?"}
    ]
}))

# COMMAND ----------

def test_demo_permissions(host, secret_scope, secret_key, vs_endpoint_name, index_name, embedding_endpoint_name = None, managed_embeddings = True):
  error = False
  CSS_REPORT = """
  <style>
  .dbdemos_install{
                      font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji,FontAwesome;
  color: #3b3b3b;
  box-shadow: 0 .15rem 1.15rem 0 rgba(58,59,69,.15)!important;
  padding: 10px 20px 20px 20px;
  margin: 10px;
  font-size: 14px !important;
  }
  .dbdemos_block{
      display: block !important;
      width: 900px;
  }
  .code {
      padding: 5px;
      border: 1px solid #e4e4e4;
      font-family: monospace;
      background-color: #f5f5f5;
      margin: 5px 0px 0px 0px;
      display: inline;
  }
  </style>"""

  def display_error(title, error, color=""):
    displayHTML(f"""{CSS_REPORT}
      <div class="dbdemos_install">
                          <h1 style="color: #eb0707">Configuration error: {title}</h1> 
                            {error}
                        </div>""")
  
  def get_email():
    try:
      return spark.sql('select current_user() as user').collect()[0]['user']
    except:
      return 'Uknown'

  def get_token_error(msg, e):
    return f"""
    {msg}<br/><br/>
    Your model will be served using Databrick Serverless endpoint and needs a Pat Token to authenticate.<br/>
    <strong> This must be saved as a secret to be accessible when the model is deployed.</strong><br/><br/>
    Here is how you can add the Pat Token as a secret available within your notebook and for the model:
    <ul>
    <li>
      first, setup the Databricks CLI on your laptop or using this cluster terminal:
      <div class="code dbdemos_block">pip install databricks-cli</div>
    </li>
    <li> 
      Configure the CLI. You'll need your workspace URL and a PAT token from your profile page
      <div class="code dbdemos_block">databricks configure</div>
    </li>  
    <li>
      Create the dbdemos scope:
      <div class="code dbdemos_block">databricks secrets create-scope dbdemos</div>
    <li>
      Save your service principal secret. It will be used by the Model Endpoint to autenticate. <br/>
      If this is a demo/test, you can use one of your PAT token.
      <div class="code dbdemos_block">databricks secrets put-secret dbdemos rag_sp_token</div>
    </li>
    <li>
      Optional - if someone else created the scope, make sure they give you read access to the secret:
      <div class="code dbdemos_block">databricks secrets put-acl dbdemos '{get_email()}' READ</div>

    </li>  
    </ul>  
    <br/>
    Detailed error trying to access the secret:
      <div class="code dbdemos_block">{e}</div>"""

  try:
    secret = dbutils.secrets.get(secret_scope, secret_key)
    secret_principal = "__UNKNOWN__"
    try:
      from databricks.sdk import WorkspaceClient
      w = WorkspaceClient(token=dbutils.secrets.get(secret_scope, secret_key), host=host)
      secret_principal = w.current_user.me().emails[0].value
    except Exception as e_sp:
      error = True
      display_error(f"Couldn't get the SP identity using the Pat Token saved in your secret", 
                    get_token_error(f"<strong>This likely means that the Pat Token saved in your secret {secret_scope}/{secret_key} is incorrect or expired. Consider replacing it.</strong>", e_sp))
      return
  except Exception as e:
    error = True
    display_error(f"We couldn't access the Pat Token saved in the secret {secret_scope}/{secret_key}", 
                  get_token_error("<strong>This likely means your secret isn't set or not accessible for your user</strong>.", e))
    return
  
  try:
    from databricks.vector_search.client import VectorSearchClient
    # update vsc client to use service principal rather than pat
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=secret, disable_notice=True)
    # vsc = VectorSearchClient(workspace_url=host, service_principal_client_id=OAuthID, service_principal_client_secret=secret_key)
    vs_index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name)
    if embedding_endpoint_name:
      if managed_embeddings:
        from langchain_community.embeddings import DatabricksEmbeddings
        results = vs_index.similarity_search(query_text='What is Apache Spark?', columns=["content"], num_results=1)
      else:
        from langchain_community.embeddings import DatabricksEmbeddings
        embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)
        embeddings = embedding_model.embed_query('What is Apache Spark?')
        results = vs_index.similarity_search(query_vector=embeddings, columns=["content"], num_results=1)

  except Exception as e:
    error = True
    vs_error = f"""
    Why are we getting this error?<br/>
    The model is using the Pat Token saved with the secret {secret_scope}/{secret_key} to access your vector search index '{index_name}' (host:{host}).<br/><br/>
    To do so, the principal owning the Pat Token must have USAGE permission on your schema and READ permission on the index.<br/>
    The principal is the one who generated the token you saved as secret: `{secret_principal}`. <br/>
    <i>Note: Production-grade deployement should to use a Service Principal ID instead.</i><br/>
    <br/>
    Here is how you can fix it:<br/><br/>
    <strong>Make sure your Service Principal has USE privileve on the schema</strong>:
    <div class="code dbdemos_block">
    spark.sql('GRANT USAGE ON CATALOG `{catalog}` TO `{secret_principal}>`');<br/>
    spark.sql('GRANT USAGE ON DATABASE `{catalog}`.`{db}` TO `{secret_principal}`');<br/>
    </div>
    <br/>
    <strong>Grant SELECT access to your SP to your index:</strong>
    <div class="code dbdemos_block">
    from databricks.sdk import WorkspaceClient<br/>
    import databricks.sdk.service.catalog as c<br/>
    WorkspaceClient().grants.update(c.SecurableType.TABLE, "{index_name}",<br/>
                                            changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="{secret_principal}")])
    </div>
    <br/>
    <strong>If this is still not working, make sure the value saved in your {secret_scope}/{secret_key} secret is your SP pat token </strong>.<br/>
    <i>Note: if you're using a shared demo workspace, please do not change the secret value if was set to a valid SP value by your admins.</i>

    <br/>
    <br/>
    Detailed error trying to access the endpoint:
    <div class="code dbdemos_block">{str(e)}</div>
    </div>
    """
    if "403" in str(e):
      display_error(f"Permission error on Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
    else:
      display_error(f"Unkown error accessing the Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
  def get_wid():
    try:
      return dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('orgId')
    except:
      return None
  if get_wid() in ["5206439413157315", "984752964297111", "1444828305810485", "2556758628403379"]:
    print(f"----------------------------\nYou are in a Shared FE workspace. Please don't override the secret value (it's set to the SP `{secret_principal}`).\n---------------------------")

  if not error:
    print('Secret and permissions seems to be properly setup, you can continue the demo!')

# COMMAND ----------

catalog = 'ang_nara_catalog'
db = 'llmops'
VECTOR_SEARCH_ENDPOINT_NAME = 'dbdoc_vs_endpoint'

index_name=f"{catalog}.{db}.databricks_pdf_documentation_self_managed_vs_index"
host = "https://e2-demo-field-eng.cloud.databricks.com"

#Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance

#Setup service principal and secret scope and key

test_demo_permissions(host, secret_scope="dbrag", secret_key="rag_pat_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en", managed_embeddings = False)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
# from langchain.chains import RetrievalQA
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbrag", "rag_pat_token")

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    # vsc = VectorSearchClient(workspace_url=host, service_principal_client_id=, service_principal_client_secret=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model, columns=["url"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?"}]}))

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?"}, 
        {"role": "assistant", "content": "The patient underwent several treatment regimens for his relapsed AML with concurrent molecular aberration recurrence after remission duration, including venetoclax combined with azacitidine and chidamide with venetoclax and azacitidine regimen. He achieved complete remission after one course of the chidamide combined regimen."}, 
        {"role": "user", "content": "What was this patients discharge summary?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a trustful assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, AI, ML, Datawarehouse, platform, API or infrastructure, Cloud administration question related to Databricks. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about patients history or treatments.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_databricks_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?"}, 
        {"role": "assistant", "content": "The patient underwent several treatment regimens for his relapsed AML with concurrent molecular aberration recurrence after remission duration, including venetoclax combined with azacitidine and chidamide with venetoclax and azacitidine regimen. He achieved complete remission after one course of the chidamide combined regimen."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
print(non_relevant_dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What treatment regimens were used for the patient's relapsed AML with concurrent molecular aberration recurrence after remission duration?"}, 
        {"role": "assistant", "content": "The patient underwent several treatment regimens for his relapsed AML with concurrent molecular aberration recurrence after remission duration, including venetoclax combined with azacitidine and chidamide with venetoclax and azacitidine regimen. He achieved complete remission after one course of the chidamide combined regimen."}, 
        {"role": "user", "content": "What was this patients discharge summary?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
print(dialog["messages"], response)

# COMMAND ----------

import cloudpickle
import langchain
import mlflow
from mlflow.models import infer_signature

catalog = 'ang_nara_catalog'
db = 'llmops'

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
