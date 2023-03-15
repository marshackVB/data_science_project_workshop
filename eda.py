# Databricks notebook source
# MAGIC %md ## Exploratory data analysis  
# MAGIC Options for interacting with your data

# COMMAND ----------

from helpers import get_current_user

# COMMAND ----------

current_user = get_current_user()

training_data_table = f"default.{current_user}_train"
print(training_data_table)

# COMMAND ----------

# MAGIC %md #### SQL

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM default.marshall_carter_train
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMP VIEW marshall_carter_train_view AS
# MAGIC SELECT * FROM default.marshall_carter_train
# MAGIC WHERE Survived == 1;
# MAGIC 
# MAGIC SELECT * FROM marshall_carter_train_view;

# COMMAND ----------

# MAGIC %md #### Pyspark

# COMMAND ----------

df = spark.table('marshall_carter_train_view')
display(df)

# COMMAND ----------

gender_counts = df.groupBy('Sex').count()
display(gender_counts)

# COMMAND ----------

# MAGIC %md #### A mix of SQL and the DataFrame syntax

# COMMAND ----------

expressions = ["Survived", "case when Age < 20 then '< 20' else '> 20' end as under_20"]

gender_counts = (spark.table(training_data_table)
                      .selectExpr(expressions)
                      .groupBy('under_20', 'Survived').count())

display(gender_counts)

# COMMAND ----------

# MAGIC %md #### Koalas  
# MAGIC 
# MAGIC The Pandas API on Spark: https://github.com/databricks/koalas
