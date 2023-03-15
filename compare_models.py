# Databricks notebook source
# MAGIC %md ## Compare models and promote best model to Model Registry

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from helpers import get_current_user

# COMMAND ----------

current_user = get_current_user()
print(current_user)

# COMMAND ----------

# MAGIC %md Retrieve the best model run

# COMMAND ----------

client = MlflowClient()

experiment_location = f'/Shared/{current_user}'
experiment_id = mlflow.get_experiment_by_name(experiment_location).experiment_id
print(experiment_id)

# COMMAND ----------

runs = client.search_runs(experiment_ids=[experiment_id],
                          order_by=['metrics.eval_f1_score DESC'], 
                          max_results=1)[0]
runs

# COMMAND ----------

best_model_run_id = runs.info.run_id
best_model_artifact_uri = runs.info.artifact_uri

print(best_model_run_id)
print(best_model_artifact_uri)

# COMMAND ----------

# MAGIC %md Register best model

# COMMAND ----------

model_registry_name = f"{current_user}_model"

registered_model = client.create_model_version(
                   name = model_registry_name,
                   source = f"{best_model_artifact_uri}/model",
                   run_id = best_model_run_id
                    )

# COMMAND ----------

# MAGIC %md Promote best model to the 'Production' stage

# COMMAND ----------

promote_to_prod = client.transition_model_version_stage(name=model_registry_name,
                                                        version = int(registered_model.version),
                                                        stage="Production",
                                                        archive_existing_versions=True)
