# Databricks notebook source
# MAGIC %md ## Raw data ingestion and transformation

# COMMAND ----------

# MAGIC %md See Ipython autoreload [documentation](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html). This allow us to edit Python modules and have the changes automatically available in subsequent notebook cell runs.

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as func
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, StructField
from helpers import get_current_user, get_spark_dataframe

# COMMAND ----------

# MAGIC %md Load the raw data from .csv format into a Spark Dataframe

# COMMAND ----------

raw_data = get_spark_dataframe()
display(raw_data)

# COMMAND ----------

# MAGIC %md Transform raw data

# COMMAND ----------

transformed_data = (raw_data.withColumn('NamePrefix', func.regexp_extract(col('Name'), '([A-Za-z]+)\.', 1))
                            .withColumn('NameSecondary_extract', func.regexp_extract(col('Name'), '\(([A-Za-z ]+)\)', 1))
                            .withColumn('TicketChars_extract', func.regexp_extract(col('Ticket'), '([A-Za-z]+)', 1))
                            .withColumn("CabinChar", func.split(col("Cabin"), '')[0])
                            .withColumn("CabinMulti_extract", func.size(func.split(col("Cabin"), ' ')))
                            .withColumn("FareRounded", func.round(col("Fare"), 0))
                    
                            .selectExpr("PassengerId",
                                        "Sex",
                                        "case when Age = 'NaN' then NULL else Age end as Age",
                                        "SibSp",
                                        "NamePrefix",
                                        "FareRounded",
                                        "CabinChar",
                                        "Embarked",
                                        "Parch",
                                        "Pclass",
                                        "case when length(NameSecondary_extract) > 0 then NameSecondary_extract else NULL end as NameSecondary",
                                        "case when length(TicketChars_extract) > 0 then upper(TicketChars_extract) else NULL end as TicketChars",
                                        "case when CabinMulti_extract < 0 then '0' else cast(CabinMulti_extract as string) end as CabinMulti",
                                        "Survived")
                   
                           .selectExpr("*",
                                       "case when NameSecondary is not NULL then '1' else '0' end as NameMultiple",
                                       "case when TicketChars is NULL then '1' else '0' end as MissingTicketChars")
                  
                           .drop("NameSecondary", "TicketChars"))

display(transformed_data)

# COMMAND ----------

# MAGIC %md Write transformed data to Delta

# COMMAND ----------

current_user = get_current_user()
print(f'Current user: {current_user}')

transformed_data.write.mode('overwrite').format('delta').saveAsTable(f'default.{current_user}_train')

display(spark.table(f"default.{current_user}_train"))
