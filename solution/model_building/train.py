# for data manipulation
import pandas as pd

# for model training and evaluation
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline          # imblearn Pipeline: supports SMOTE step
from imblearn.over_sampling import SMOTE

# for model evaluation metrics
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

# =========================================================================
# SELECTED MODEL: Bagging (best validation F1 on the breakdown class)
# Operating threshold 0.32: chosen on the validation set as the point that
# guarantees >= 95% breakdown recall with the highest possible precision.
# =========================================================================

bagging_best_params = {
    "classifier__n_estimators": 200,
    "classifier__max_samples": 1.0,
    "classifier__max_features": 0.7,
}

AGGRESSIVE_THRESHOLD = 0.32   # selected on validation (recall >= 0.95)

# Pipeline mirrors the notebook: StandardScaler -> SMOTE -> Bagging
bagging_pipeline = Pipeline([
    ("preprocessor", StandardScaler()),
    ("smote", SMOTE(random_state=1)),
    ("classifier", BaggingClassifier(
        random_state=1,
        **{k.replace("classifier__", ""): v for k, v in bagging_best_params.items()}
    )),
])


with mlflow.start_run():
    # Fit the Bagging pipeline with the tuned parameters
    bagging_pipeline.fit(X_train_aligned, y_train)

    # Log parameters, including the operating threshold
    mlflow.log_params(bagging_best_params)
    mlflow.log_param("operating_threshold", AGGRESSIVE_THRESHOLD)
    mlflow.log_param("model_type", "BaggingClassifier")

    # --- Test metrics at the DEFAULT 0.50 threshold (for reference) ---
    probabilities = bagging_pipeline.predict_proba(X_test_aligned)[:, 1]
    y_pred_test_default = (probabilities >= 0.50).astype(int)

    # --- Test metrics at the SELECTED recall-first threshold ---
    y_pred_test_aggressive = (probabilities >= AGGRESSIVE_THRESHOLD).astype(int)

    test_metrics = {
        # default threshold
        "test_accuracy_default": accuracy_score(y_test, y_pred_test_default),
        "test_precision_class1_default": precision_score(y_test, y_pred_test_default, pos_label=1),
        "test_recall_class1_default": recall_score(y_test, y_pred_test_default, pos_label=1),
        "test_f1_class1_default": f1_score(y_test, y_pred_test_default, pos_label=1),
        # selected recall-first threshold
        "test_accuracy_aggressive": accuracy_score(y_test, y_pred_test_aggressive),
        "test_precision_class1_aggressive": precision_score(y_test, y_pred_test_aggressive, pos_label=1),
        "test_recall_class1_aggressive": recall_score(y_test, y_pred_test_aggressive, pos_label=1),
        "test_f1_class1_aggressive": f1_score(y_test, y_pred_test_aggressive, pos_label=1),
    }
    mlflow.log_metrics(test_metrics)

    print("Test metrics logged to MLflow:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save the model locally
    model_path = "bagging_classifier_model.joblib"
    joblib.dump(bagging_pipeline, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Bagging Model saved as artifact at: {model_path}")

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

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="bagging_classifier_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
