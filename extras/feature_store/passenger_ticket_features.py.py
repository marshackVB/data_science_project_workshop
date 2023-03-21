# Databricks notebook source
# MAGIC %md ### Feature engineering logic for ticket features

# COMMAND ----------

from pyspark.sql.functions import col
import pyspark.sql.functions as func
from databricks.feature_store import FeatureStoreClient
from databricks.feature_store import feature_table
from helpers import get_current_user, get_spark_dataframe

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

current_user = get_current_user()
print(f'Current user: {current_user}')

# COMMAND ----------

raw_data = get_spark_dataframe('../../titanic_train.csv')
display(raw_data)

# COMMAND ----------

# MAGIC %md Define feature transformation logic

# COMMAND ----------

def compute_passenger_ticket_features(df):
  
            # Extract characters of ticket if they exist
  return (df.withColumn('TicketChars_extract', func.regexp_extract(col('Ticket'), '([A-Za-z]+)', 1))
             .selectExpr("*", "case when length(TicketChars_extract) > 0 then upper(TicketChars_extract) else NULL end as TicketChars")
             .drop("TicketChars_extract")
          
            # Extract the Cabin character
             .withColumn("CabinChar", func.split(col("Cabin"), '')[0])
          
            # Indicate if multiple Cabins are present
             .withColumn("CabinMulti_extract", func.size(func.split(col("Cabin"), ' ')))
             .selectExpr("*", "case when CabinMulti_extract < 0 then '0' else cast(CabinMulti_extract as string) end as CabinMulti")
             .drop("CabinMulti_extract")
          
            # Round the Fare column
             .withColumn("FareRounded", func.round(col("Fare"), 0))
         
             .drop('Ticket', 'Cabin'))

# COMMAND ----------

# MAGIC %md Apply transformation logic to source table

# COMMAND ----------

passenger_ticket_features = compute_passenger_ticket_features(raw_data)

# COMMAND ----------

display(passenger_ticket_features)

# COMMAND ----------

# MAGIC %md Create an entry in the feature store if one does not exist

# COMMAND ----------

feature_table_name = f'default.ticket_features_{current_user}'

# If the feature table has already been created, no need to recreate
try:
  fs.get_table(feature_table_name)
  print("Feature table entry already exists")
  pass
  
except Exception:
  fs.create_table(name = feature_table_name,
                          primary_keys = 'PassengerId',
                          schema = passenger_ticket_features.schema,
                          description = 'Ticket-related features for Titanic passengers')

# COMMAND ----------

fs.write_table(
  
  name= feature_table_name,
  df = passenger_ticket_features,
  mode = 'merge'
  
  )

# COMMAND ----------

# MAGIC %md To drop the feature table, delete the underlying Delta table; the table must also be deleted in the feature store UI

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC -- DROP TABLE IF EXISTS default.ticket_features;
