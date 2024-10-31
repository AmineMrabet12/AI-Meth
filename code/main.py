import pandas as pd
import os
from data_preparation import prepare
from preprocessing import preprocess
from train import train
from evaluation import evaluate
import mlflow

# Configure MLflow tracking and experiment
mlflow.set_tracking_uri("mlruns")
experiment_name = "AI-Meth Xp-V1"
mlflow.set_experiment(experiment_name)

# Load and prepare data
df = pd.read_excel('data/E Commerce Dataset.xlsx', sheet_name='E Comm')

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
model_paths = [os.path.join('models', file_name) for file_name in os.listdir('models')]

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
