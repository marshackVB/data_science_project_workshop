# Databricks notebook source
# MAGIC %md ### Creating a custom MLflow model. 
# MAGIC 
# MAGIC See the [documentation](https://mlflow.org/docs/latest/models.html#model-customization)  
# MAGIC 
# MAGIC Use case:
# MAGIC   - You want to log a model that is not part of a build-in [MLflow model flavor](https://mlflow.org/docs/latest/models.html#built-in-model-flavors).
# MAGIC   - You want to alter the behavior of an MLflow model flavor, for instance, by adding in pre and/or post processing.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from helpers import get_current_user, get_or_create_experiment

# COMMAND ----------

mlflow.autolog(disable=True)

# COMMAND ----------

current_user = get_current_user()
print(f'Current user: {current_user}')

experiment_location = f'/Shared/{current_user}'
dbutils.fs.mkdirs(experiment_location)

get_or_create_experiment(experiment_location)

# COMMAND ----------

# MAGIC %md ### Simple example  
# MAGIC Create a class that inherits from **mlflow.pyfunc.PythonModel**; override the parent class' predict method.

# COMMAND ----------

class AddN(mlflow.pyfunc.PythonModel):
  """
  Create a model that give a row of data, will add the value, n, to 
  each column value
  """
  def __init__(self, n: int):
      self.n = n

  def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
      return model_input.apply(lambda column: column + self.n)

# COMMAND ----------

# MAGIC %md Log the model to MLflow

# COMMAND ----------

with mlflow.start_run(run_name='add_n_model') as run:
  
  run_id = run.info.run_id
  
  add5_model = AddN(n=5)
  
  mlflow.pyfunc.log_model(artifact_path="model", 
                          python_model=add5_model)
  
  print(f"Simple model run_id: {run_id}")

# COMMAND ----------

# MAGIC %md Load the model and perform inference

# COMMAND ----------

model_input = pd.DataFrame([range(10)])
model_input.head()

# COMMAND ----------

logged_model = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)

model_output = loaded_model.predict(model_input)
model_output.head()

# COMMAND ----------

# MAGIC %md ### XGBoost example

# COMMAND ----------

# MAGIC %md Load sample data

# COMMAND ----------

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(x_train, label=y_train)

num_obs = 4

print("Training samples")
print(x_train[:num_obs])

print("\n Training labels")
print(y_train[:num_obs])

# COMMAND ----------

# MAGIC %md Train an XGBoost model and save it to the driver node

# COMMAND ----------

xgb_model = xgb.train(params={"max_depth": 10}, dtrain=dtrain, num_boost_round=10)
xgb_model_path = "xgb_model.pth"
xgb_model.save_model(xgb_model_path)
