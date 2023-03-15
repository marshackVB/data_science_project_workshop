import mlflow
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


def get_current_user():
  """Get the current notebook user"""
  return dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0].replace('.', '_')


def get_or_create_experiment(experiment_location):
  """Create an Mlflow experiment if one does not exist"""
  
  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)


def create_registry_entry(model_registry_name):
  """Create an MLflow Model Registry entry for the project if 
  one does not exist
  """
  
  client = MlflowClient()
  
  try:
    client.get_registered_model(model_registry_name)
    print(" Registered model already exists")
  except:
    client.create_registered_model(model_registry_name)


def get_model_info(model_name, stage):
  """Retrieve information associated with an MLflow run"""
  
  client = MlflowClient()
  
  run_info = [run for run in client.search_model_versions(f"name='{model_name}'") 
                  if run.current_stage == stage][0]
  return run_info


def get_run_id(model_name, stage='Production'):
  """Retrieve a model's run id"""
  
  client = MlflowClient()
  
  return get_model_info(model_name, stage).run_id


def get_pipeline():
  """
  Return a scikit-learn ColumnTranformer that performs feature pre-processing
  """
  categorical_vars = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
  numeric_vars = ['Age', 'FareRounded']
  binary_vars = ['NameMultiple', 'MissingTicketChars']

  # Create the a pre-processing and modleing pipeline
  binary_transform = make_pipeline(SimpleImputer(strategy = 'constant', fill_value = 'missing'))

  numeric_transform = make_pipeline(SimpleImputer(strategy = 'most_frequent'))

  categorical_transform = make_pipeline(SimpleImputer(missing_values = None, strategy = 'constant', fill_value = 'missing'), 
                                        OneHotEncoder(handle_unknown="ignore"))

  transformer = ColumnTransformer([('categorial_vars', categorical_transform, categorical_vars),
                                    ('numeric_vars', numeric_transform, numeric_vars),
                                    ('binary_vars', binary_transform, binary_vars)],
                                    remainder = 'drop')

  return transformer