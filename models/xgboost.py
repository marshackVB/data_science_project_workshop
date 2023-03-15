# Databricks notebook source
# MAGIC 
# MAGIC %md ## XGBoost model training workflow

# COMMAND ----------

import mlflow
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from helpers import get_current_user, get_or_create_experiment, create_registry_entry, get_pipeline

# COMMAND ----------

current_user = get_current_user()
print(current_user)

# COMMAND ----------

# MAGIC %md Establish reference to MLflow experiment

# COMMAND ----------

experiment_location = f'/Shared/{current_user}'
dbutils.fs.mkdirs(experiment_location)

get_or_create_experiment(experiment_location)

# COMMAND ----------

dbutils.fs.ls(f'/Shared/{current_user}/experiments')

# COMMAND ----------

# MAGIC %md Ingest data

# COMMAND ----------

training_df = spark.table(f"default.{current_user}_train").toPandas()
training_df.head()

# COMMAND ----------

# MAGIC %md Train model and log to mlflow

# COMMAND ----------

with mlflow.start_run(run_name='xgboost') as run:
  
  mlflow.autolog(log_models=False)
  
  run_id = run.info.run_id
  
  label = 'Survived'
  features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

  X_train, X_test, y_train, y_test = train_test_split(training_df[features], training_df[label], test_size=0.25, random_state=123, shuffle=True)

  preprocessing_pipeline = get_pipeline()

  model = xgb.XGBClassifier(n_estimators = 25, use_label_encoder=False)

  classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])

  classification_pipeline.fit(X_train, y_train)
  
  train_metrics = mlflow.sklearn.eval_and_log_metrics(classification_pipeline, X_train, y_train, prefix="train_")
  eval_metrics = mlflow.sklearn.eval_and_log_metrics(classification_pipeline, X_test, y_test, prefix="eval_")
  
  mlflow.autolog(log_input_examples=True,
                 log_model_signatures=True,
                 log_models=True)
  
  # Train final model on all data
  classification_pipeline.fit(training_df[features], training_df[label])

# COMMAND ----------

# MAGIC %md Create registry entry if one does not exist

# COMMAND ----------

create_registry_entry(f"{current_user}_model")
