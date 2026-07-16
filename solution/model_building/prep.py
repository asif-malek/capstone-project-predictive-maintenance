# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from imblearn.over_sampling import SMOTE

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/asifaddicted/vehicle-breakdown-prediction/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

df.columns = df.columns.str.replace(' ', '_')


# Define target variable
target_col = 'Engine_Condition'




# =========================================================================
# STEP 1 & 2: ENGINE FEATURE ENGINEERING
# =========================================================================
# Create a copy of your dataframe to protect original data
df_engineered = df.copy()

# 1. Stress Ratio: High RPM combined with low oil pressure means extreme friction
df_engineered["RPM_to_OilPres_Ratio"] = (
    df_engineered["Engine_rpm"] / (df_engineered["Lub_oil_pressure"] + 0.001)
)

# 2. Temperature Divergence: Oil and coolant should heat up together. Large gaps imply cooling failure.
df_engineered["Temp_Divergence"] = abs(
    df_engineered["lub_oil_temp"] - df_engineered["Coolant_temp"]
)

# 3. Total Pressure Load: Summing up all system pressures to catch overall vacuum/burst spikes
df_engineered["Total_Pressure"] = (
    df_engineered["Lub_oil_pressure"]
    + df_engineered["Fuel_pressure"]
    + df_engineered["Coolant_pressure"]
)

# Separate features and target
X = df_engineered.drop(columns=["Engine_Condition"])
y = df_engineered["Engine_Condition"]

# =========================================================================
# STEP 3: TRAIN / VAL / TEST SPLIT WITH STRATIFICATION
# =========================================================================
# We use stratify=y to ensure the exact same breakdown percentage exists in all splits
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=1, stratify=y_temp # Changed stratify=y to stratify=y_temp
)

# =========================================================================
# STEP 4: BALANCE THE TRAINING DATA ONLY
# =========================================================================
# SMOTE generates synthetic healthy samples so the model treats both outcomes equally
smote = SMOTE(random_state=1)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"New Training Shape: {X_train.shape}")
print(f"Balanced Breakdown Counts:\n{y_train.value_counts()}")



X_train.to_csv("X_train.csv",index=False)
X_val.to_csv("X_val.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_val.to_csv("y_val.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_test.to_csv("y_test.csv",index=False)


files = ["X_train.csv","X_val.csv","y_train.csv","y_val.csv", "X_test.csv", "y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="asifaddicted/vehicle-breakdown-prediction",
        repo_type="dataset",
    )
