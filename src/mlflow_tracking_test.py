import mlflow

mlflow.set_experiment("Insurance_Fraud_Detection")

with mlflow.start_run():

    mlflow.log_param("model", "RandomForest")

    mlflow.log_metric("accuracy", 0.986)

    mlflow.log_metric("roc_auc", 0.9988)

    print("MLflow Run Logged Successfully")