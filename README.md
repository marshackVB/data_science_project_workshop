## Data Science Workshop Project 

Required cluster runtime version: **12.2 LTS ML**. Each participant should provision his/her own single-node cluster using the required runtime. Clone this repository into a Databricks Repo.

Notebook run order:
   1.  **etl**: Load raw data and create a Delta table of features
   2.  **eda**: Compare/contrast Spark SQL and the DataFrame API.
   3.  **models/xgboost**: Train an XGBoost model and log to MLflow.
   4.  **models/random_forest_hyperopt**: Train a Random Forest model with hyperparameter tuning and MLflow logging.
   5.  **compare_models**: Choose the best model and register it in the Model Registry.
   6.  **score**: Load the production model from the Model Registry and perform inference.
   7.  **No notebook**: Follow along with instructor: Deploy the production model as a Rest API.
   8.  **No notebook**: Follow along with instructor: Create and run a multi-task job via the Databricks Jobs UI
   9.  **model_registry_webhook**: Watch instructor: Triggering activities base on Model Registry events.
  10.  **No notebook**: Follow along with instructor: Auto ML, training and comparing models automatically.
 


Extras if time permits:  
 - **extras/custom_mlflow_model**: Creating and logging your own, custom MLflow model.
 - **extas/feature_store**: Integrating the Databricks Feature Store into the model training and inference workflows.  
  Notebook run order:  
    1. passenger_demographic_features
    2. passenger_ticket_features
    3. fit_model
    4. model_inference