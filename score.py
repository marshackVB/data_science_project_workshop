# Databricks notebook source
# MAGIC %md ## Load the Production model from the Registry and perform inference

# COMMAND ----------

import mlflow
from helpers import get_current_user, get_run_id

# COMMAND ----------

current_user = get_current_user()
print(current_user)

# COMMAND ----------

production_run_id = get_run_id(f"{current_user}_model")
logged_model = f'runs:/{production_run_id}/model'

print(logged_model)

# COMMAND ----------

# MAGIC %md Perform inference directly on a Spark Dataframe

# COMMAND ----------

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

spark_df = spark.table(f"default.{current_user}_train")
predictions = spark_df.withColumn('predictions', loaded_model())

display(predictions)

# COMMAND ----------

predictions.write.mode('overwrite').format('delta').saveAsTable(f"default.{current_user}_predictions")
