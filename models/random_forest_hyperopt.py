# Databricks notebook source
# MAGIC %md ## Random Forest training workflow

# COMMAND ----------

from collections import OrderedDict
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
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

# MAGIC %md Parameter tuning using Hyperopt  
# MAGIC  - Hyperopt [documentation](http://hyperopt.github.io/hyperopt/)  
# MAGIC  - Tuning best practices [blog](https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html)

# COMMAND ----------

# Disable auto-logging temporarily
mlflow.autolog(disable=True)

label = 'Survived'
features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

X_train, X_val, y_train, y_val = train_test_split(training_df[features], training_df[label], test_size=0.25, random_state=123, shuffle=True)

preprocessing_pipeline = get_pipeline()

# Specify Hyperopt search space
space = {'n_estimators': hp.quniform('n_estimators', 20, 100, 1),
         'max_features': hp.uniform('max_features', 0.5, 0.9)}

# Define objective function for Hyperopt to minimize
def objective(params):
  """
  Accepts a collection of hyperparameter values to test. Trains a model
  using these hyperparameters and generates fit statics on the validation
  dataset.
  
  Hyperopt will continue to test hyperparameter combinations until its 
  object (minimizing 1 - f1 score) cannot be improved.
  """
   
  params['n_estimators'] = int(params['n_estimators'])

  model = RandomForestClassifier(**params)  

  classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])

  classification_pipeline.fit(X_train, y_train)

  precision_eval, recall_eval, f1_eval, _ = precision_recall_fscore_support(y_val, 
                                                                            classification_pipeline.predict(X_val), 
                                                                            average='weighted')
  digits = 3
  metrics = OrderedDict()
  metrics["eval_precision"]= round(precision_eval, digits)
  metrics["eval_recall"] =   round(recall_eval, digits)
  metrics["eval_f1"] =       round(f1_eval, digits)

  return {'loss': 1 - f1_eval, 'status': STATUS_OK, 'metrics': metrics}
    
trials = Trials()

best_parameters = fmin(fn=objective, 
                       space=space, 
                       algo=tpe.suggest,
                       max_evals=100, 
                       trials=trials, 
                       rstate=np.random.default_rng(50),
                       early_stop_fn=no_progress_loss(iteration_stop_count=25, percent_increase=0.5))

# COMMAND ----------

# MAGIC %md View validation statistics for the best model

# COMMAND ----------

trials.best_trial['result']['metrics']

# COMMAND ----------

# MAGIC %md View the best model hyperparameter values

# COMMAND ----------

best_parameters

# COMMAND ----------

# MAGIC %md Format the hyperparmater values from Hyperopt values to be compatible with scikit-learn

# COMMAND ----------

final_model_parameters = {}

for parameter, value in best_parameters.items():
  if parameter in ['n_estimators']:
    final_model_parameters[parameter] = int(value)
  else:
    final_model_parameters[parameter] = value
    
    
print(final_model_parameters)

# COMMAND ----------

# MAGIC %md Train the final model and log to mlflow

# COMMAND ----------

with mlflow.start_run(run_name='random_forest') as run:
  
  run_id = run.info.run_id
  
  mlflow.autolog(log_input_examples=True,
                 log_model_signatures=True,
                 log_models=True,
                 silent=True)
  
  label = 'Survived'
  features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

  X_train, X_val, y_train, y_val = train_test_split(training_df[features], training_df[label], test_size=0.25, random_state=123, shuffle=True)

  preprocessing_pipeline = get_pipeline()

  model = RandomForestClassifier(**final_model_parameters)  

  classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])

  classification_pipeline.fit(X_train, y_train)
  
  logged_model = f'runs:/{run_id}/model'
  eval_features_and_labels = pd.concat([X_val, y_val], axis=1)
  
  mlflow.evaluate(logged_model, 
                  data=eval_features_and_labels, 
                  targets="Survived", 
                  model_type="classifier")
  
  print(f"Training model with run id: {run_id}")
