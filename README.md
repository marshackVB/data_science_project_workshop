## Example Data Science project workshop. 

Required cluster runtime version: 12.2 LTS ML

Notebook run order:
  1.  etl
  2.  eda
  3.  models/xgboost
  4.  models/random_forest_hyperopt
  5.  compare_models
  6.  score
  7.  No notebook: Create and run multi-task job via Databricks Jobs UI
  8.  No notebook: Deploy model as Rest API via Model Registry UI
  9.  model_registry_webhook (follow along with instructor)


Extras if time permits:  
 - extras/custom_mlflow_model: creating and logging your own, custom MLflow model.
 - extas/feature_store: integrating the Databricks Feature Store into the model training and inference process.  
  Notebook run order:  
    1. passenger_demographic_features
    2. passenger_ticket_features
    3. fit_model
    4. model_inference