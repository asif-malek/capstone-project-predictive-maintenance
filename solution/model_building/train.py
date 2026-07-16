# for data manipulation
import pandas as pd

# for model training, tuning, and evaluation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# for model training, tuning, and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow




login(token=os.environ.get("HF_TOKEN"))

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

X_train_path = "hf://datasets/asifaddicted/vehicle-breakdown-prediction/X_train.csv"
X_val_path = "hf://datasets/asifaddicted/vehicle-breakdown-prediction/X_val.csv"
y_train_path = "hf://datasets/asifaddicted/vehicle-breakdown-prediction/y_train.csv"
y_val_path = "hf://datasets/asifaddicted/vehicle-breakdown-prediction/y_val.csv"
X_test_path = "hf://datasets/asifaddicted/vehicle-breakdown-prediction/X_test.csv"
y_test_path = "hf://datasets/asifaddicted/vehicle-breakdown-prediction/y_test.csv"

X_train = pd.read_csv(X_train_path)
X_val = pd.read_csv(X_val_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).squeeze() # Ensure y_train is a Series
y_val = pd.read_csv(y_val_path).squeeze()   # Ensure y_val is a Series
y_test = pd.read_csv(y_test_path).squeeze() # Ensure y_test is a Series

# Align columns to prevent feature mismatch errors
X_train_aligned = X_train
X_test_aligned = X_test.reindex(columns=X_train_aligned.columns, fill_value=0)


# Best parameter chosen for AdaBoost tuning (from previous RandomizedSearchCV)
adaboost_best_params = {'classifier__n_estimators': 100, 'classifier__learning_rate': 0.1}

# Create the preprocessor (StandardScaler)
preprocessor = StandardScaler()

# Create the AdaBoost pipeline
adaboost_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(random_state=1, **{k.replace('classifier__', ''): v for k, v in adaboost_best_params.items()}))
])


with mlflow.start_run():
    # Directly fit the AdaBoost pipeline with the pre-defined best parameters
    adaboost_pipeline.fit(X_train_aligned, y_train)

    # Log the parameters used for this run
    # Parameters from adaboost_best_params are logged as they were used directly
    mlflow.log_params(adaboost_best_params)

    # Predictions on training set (default 0.5 threshold)
    y_pred_train = adaboost_pipeline.predict(X_train_aligned)

    # --- Aggressive Monitoring (0.30 threshold) metrics for test set ---
    probabilities = adaboost_pipeline.predict_proba(X_test_aligned)[:, 1]
    y_pred_test_aggressive = (probabilities >= 0.30).astype(int)

    # Metrics for training set (default 0.5 threshold)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision_class1 = precision_score(y_train, y_pred_train, pos_label=1, average='binary')
    train_recall_class1 = recall_score(y_train, y_pred_train, pos_label=1, average='binary')
    train_f1_class1 = f1_score(y_train, y_pred_train, pos_label=1, average='binary')

    # Metrics for test set (aggressive 0.30 threshold)
    test_accuracy_aggressive = accuracy_score(y_test, y_pred_test_aggressive)
    test_precision_class1_aggressive = precision_score(y_test, y_pred_test_aggressive, pos_label=1, average='binary')
    test_recall_class1_aggressive = recall_score(y_test, y_pred_test_aggressive, pos_label=1, average='binary')
    test_f1_class1_aggressive = f1_score(y_test, y_pred_test_aggressive, pos_label=1, average='binary')

    # Log aggressive monitoring metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "train_precision_class1": train_precision_class1,
        "train_recall_class1": train_recall_class1,
        "train_f1_class1": train_f1_class1,
        "test_accuracy_aggressive_threshold_0.30": test_accuracy_aggressive,
        "test_precision_class1_aggressive_threshold_0.30": test_precision_class1_aggressive,
        "test_recall_class1_aggressive_threshold_0.30": test_recall_class1_aggressive,
        "test_f1_class1_aggressive_threshold_0.30": test_f1_class1_aggressive
    })

    # Save the model locally
    model_path = "adaboost_classifier_model.joblib"
    joblib.dump(adaboost_pipeline, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"AdaBoost Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "asifaddicted/vehicle-breakdown-prediction-model"  # separate model repo
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj=model_path, # Corrected filename
        path_in_repo="adaboost_classifier_model.joblib", # Corrected filename
        repo_id=repo_id,
        repo_type=repo_type,
    )
