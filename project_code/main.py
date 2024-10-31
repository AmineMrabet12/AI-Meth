import pandas as pd
import os
from data_preparation import prepare
from preprocessing import preprocess
from train import train
from evaluation import evaluate
import mlflow

"""
This script automates the entire machine learning pipeline for an E-commerce dataset, 
including data preparation, preprocessing, model training, evaluation, and result reporting.
The pipeline is tracked and logged using MLflow for experiment management.

Workflow:
1. Configures MLflow with a specified experiment name.
2. Loads and prepares the data from an Excel file.
3. Preprocesses the data by encoding categorical columns and scaling features.
4. Trains multiple machine learning models.
5. Evaluates the trained models and logs the results.
6. Summarizes evaluation metrics and displays them in a sorted DataFrame.

Modules Used:
    - `prepare` from `data_preparation`: Cleans and preprocesses the data.
    - `preprocess` from `preprocessing`: Encodes categorical columns and scales features.
    - `train` from `train`: Trains specified machine learning models on the dataset.
    - `evaluate` from `evaluation`: Calculates evaluation metrics for each model.

MLflow Configuration:
    - Sets tracking URI and experiment name for MLflow.
    - Logs models, metrics, and artifacts during training and evaluation.

Output:
    - Prints status updates and the resulting evaluation metrics for each model.
    - Displays a sorted summary DataFrame of model performance based on 'ROC AUC'.

Dependencies:
    - Requires `mlflow`, `pandas`, and `sklearn`.

Example Usage:
    Run this script to train and evaluate models on the specified E-commerce dataset:
        $ python main_script.py
"""

# Configure MLflow tracking and experiment
mlflow.set_tracking_uri("mlruns")
experiment_name = "AI-Meth Xp-V1"
mlflow.set_experiment(experiment_name)

# Load and prepare data
df = pd.read_excel('../data/E Commerce Dataset.xlsx', sheet_name='E Comm')

print('################### Starting Data Preparation ###################')
df, col_to_encode = prepare(df)
print('...DONE')

print('################### Starting Data Preprocessing #################')
X_train, X_test, y_train, y_test = preprocess(df, col_to_encode)
print('...DONE')
# Train models
print('################### Starting training ###########################')

train(X_train, y_train)
print('...DONE')

# Collect model paths for evaluation
model_paths = [os.path.join('../models', file_name) for file_name in os.listdir('../models')]

# Evaluate models
print('################### Starting Evaluation #########################')

results = evaluate(X_train, X_test, y_test, model_paths)
print('...DONE')

# Print summary results
print()
print(results)
print()

print('################### Results DataFrame ###########################')
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='ROC AUC', ascending=False).reset_index(drop=True)
results_df[['Accuracy', 'F1 Score', 'ROC AUC']] = results_df[['Accuracy', 'F1 Score', 'ROC AUC']].round(2)

print(results_df)
