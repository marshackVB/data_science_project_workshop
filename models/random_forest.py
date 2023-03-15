# Databricks notebook source
# MAGIC %md ## Random Forest training workflow

# COMMAND ----------

from collections import OrderedDict
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
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

# MAGIC %md Ingest data

# COMMAND ----------

training_df = spark.table(f"default.{current_user}_train").toPandas()
training_df.head()

# COMMAND ----------

mlflow.autolog(disable=True)

label = 'Survived'
features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

X_train, X_test, y_train, y_test = train_test_split(training_df[features], training_df[label], test_size=0.25, random_state=123, shuffle=True)

preprocessing_pipeline = get_pipeline()

model = RandomForestClassifier() 

classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])

classification_pipeline.fit(X_train, y_train)

precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_test, 
                                                                             classification_pipeline.predict(X_test), 
                                                                             average='weighted')

# COMMAND ----------

# MAGIC %md Parameter tuning using Hyperopt

# COMMAND ----------

mlflow.autolog(disable=True)

label = 'Survived'
features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

X_train, X_test, y_train, y_test = train_test_split(training_df[features], training_df[label], test_size=0.25, random_state=123, shuffle=True)

preprocessing_pipeline = get_pipeline()

# Hyperopt search space
space = {'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
         'max_features': hp.uniform('max_features', 0.5, 1.0)}

# Define objective function to minimize
def objective(params):
   
    params['n_estimators'] = int(params['n_estimators'])
  
    model = RandomForestClassifier(**params)  

    classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])
    
    classification_pipeline.fit(X_train, y_train)
    
    precision_eval, recall_eval, f1_eval, _ = precision_recall_fscore_support(y_test, 
                                                                              classification_pipeline.predict(X_test), 
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

# Best model validation statistics
trials.best_trial['result']['metrics']

# COMMAND ----------

# Best model parameter values
best_parameters

# COMMAND ----------

# Format best parameters for scikit-learn
final_model_parameters = {}

for parameter, value in best_parameters.items():
  if parameter in ['n_estimators']:
    final_model_parameters[parameter] = int(value)
  else:
    final_model_parameters[parameter] = value
    
    
print(final_model_parameters)

# COMMAND ----------

# MAGIC %md Train model and log to mlflow

# COMMAND ----------

mlflow.autolog(log_models=False)

with mlflow.start_run(run_name='random_forest') as run:
  
  run_id = run.info.run_id
  
  label = 'Survived'
  features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

  X_train, X_test, y_train, y_test = train_test_split(training_df[features], training_df[label], test_size=0.25, random_state=123, shuffle=True)

  preprocessing_pipeline = get_pipeline()

  model = RandomForestClassifier(**final_model_parameters)  

  classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])

  classification_pipeline.fit(X_train, y_train)
  
  train_metrics = mlflow.sklearn.eval_and_log_metrics(classification_pipeline, X_train, y_train, prefix="train_")
  eval_metrics = mlflow.sklearn.eval_and_log_metrics(classification_pipeline, X_test, y_test, prefix="eval_")
  
  mlflow.autolog(log_input_examples=True,
                 log_model_signatures=True,
                 log_models=True)
  
  classification_pipeline.fit(training_df[features], training_df[label])

# COMMAND ----------

# MAGIC %md Create registry entry if one does not exist

# COMMAND ----------

create_registry_entry(f"{current_user}_model")
