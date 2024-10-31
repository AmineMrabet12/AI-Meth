import pandas as pd
import os
from data_preparation import prepare
from preprocessing import preprocess
from train import train
from evaluation import evaluate
import mlflow
import mlflow.sklearn


df = pd.read_excel('data/E Commerce Dataset.xlsx', sheet_name='E Comm')

df, col_to_encode = prepare(df)

X_train, X_test, y_train, y_test = preprocess(df, col_to_encode)

models = train(X_train, y_train)

model_paths = []

for file_name in os.listdir('models'):
    model_paths.append('models/'+file_name)

experiment_name = "AI-Meth 1st Xp"
mlflow.set_experiment(experiment_name)

mlflow.set_tracking_uri("mlruns")

results = evaluate(X_train, X_test, y_test, model_paths)

print(results)
