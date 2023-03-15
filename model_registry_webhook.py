# Databricks notebook source
# MAGIC %md ## Create a Model Registry webhook  
# MAGIC https://docs.databricks.com/mlflow/model-registry-webhooks.html
# MAGIC 
# MAGIC https://docs.databricks.com/mlflow/model-registry-webhooks.html#job-registry-webhook-example-workflow

# COMMAND ----------

# MAGIC %pip install databricks-registry-webhooks

# COMMAND ----------

from databricks_registry_webhooks import RegistryWebhooksClient, JobSpec
from helpers import get_current_user

# COMMAND ----------

# MAGIC %run ./tmp

# COMMAND ----------

current_user = get_current_user()
model_registry_name = f"{current_user}_model"
print(model_registry_name)

# COMMAND ----------

# MAGIC %md Create a webhook to trigger an existing Databricks job

# COMMAND ----------

access_token = access_token
# https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#job/104358498399533
job_id = "104358498399533"

job_spec = JobSpec(
  job_id=job_id,
  access_token=access_token
)

job_webhook = RegistryWebhooksClient().create_webhook(
  model_name=model_registry_name,
  events=["MODEL_VERSION_TRANSITIONED_TO_PRODUCTION"],
  job_spec=job_spec,
  description="Job webhook trigger",
  status="TEST_MODE"
)

# COMMAND ----------

# MAGIC %md List webhooks

# COMMAND ----------

RegistryWebhooksClient().list_webhooks(model_name=model_registry_name)

# COMMAND ----------

# MAGIC %md Test the webhook

# COMMAND ----------

RegistryWebhooksClient().test_webhook(id=job_webhook.id)

# COMMAND ----------

# MAGIC %md Make the webhook active  
# MAGIC Model Registry [link](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/models/marshall_carter_model)

# COMMAND ----------

RegistryWebhooksClient().update_webhook(
  id=job_webhook.id,
  status="ACTIVE"
)

# COMMAND ----------

# MAGIC %md Delete the webhook

# COMMAND ----------

RegistryWebhooksClient().delete_webhook(
  id=job_webhook.id
)
