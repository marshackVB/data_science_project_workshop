# Databricks notebook source
# MAGIC 
# MAGIC %md ## XGBoost model training workflow

# COMMAND ----------

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from helpers import get_current_user, get_or_create_experiment, create_registry_entry, get_pipeline

# COMMAND ----------

current_user = get_current_user()
print(f'Current user: {current_user}')

# COMMAND ----------

# MAGIC %md Establish reference to MLflow experiment

# COMMAND ----------

experiment_location = f'/Shared/{current_user}'
dbutils.fs.mkdirs(experiment_location)

get_or_create_experiment(experiment_location)

# COMMAND ----------

# MAGIC %md Ingest data

# COMMAND ----------

training_df = spark.table(f"default.{current_user}_train").toPandas()
training_df.head()

# COMMAND ----------

# MAGIC %md Train model and log to mlflow. See the MLflow [documentation](https://mlflow.org/docs/latest/index.html).  
# MAGIC   - [Auto logging](https://mlflow.org/docs/latest/tracking.html#automatic-logging)  
# MAGIC   - [Model evaluation](https://mlflow.org/docs/latest/models.html#model-evaluation)

# COMMAND ----------

# Start an MLflow Experiment Run, which will be recorded in the
# MLflow experiment for this project
with mlflow.start_run(run_name='xgboost') as run:
  
  # Capture run id for later use
  run_id = run.info.run_id
  
  # Enable MLflow auto logging
  mlflow.autolog(log_input_examples=True,
                 log_model_signatures=True,
                 log_models=True,
                 silent=True)
  
  # Split features into train and validation
  label = 'Survived'
  features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

  X_train, X_val, y_train, y_val = train_test_split(training_df[features], training_df[label], test_size=0.25, random_state=123, shuffle=True)

  # Load the scikit-learn pre-processing pipeline
  preprocessing_pipeline = get_pipeline()

  # Create a model instance
  model = xgb.XGBClassifier(n_estimators = 25)

  # Add the model instance as a step in the pre-processing pipeline
  classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])

  # Perform pre-processing and train the model
  classification_pipeline.fit(X_train, y_train)
  
  # Evaluate the model on the validation dataset
  logged_model = f'runs:/{run_id}/model'
  eval_features_and_labels = pd.concat([X_val, y_val], axis=1)
  
  mlflow.evaluate(logged_model, 
                  data=eval_features_and_labels, 
                  targets="Survived", 
                  model_type="classifier")
  
  print(f"Training model with run id: {run_id}")
