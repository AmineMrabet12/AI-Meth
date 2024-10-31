import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import load
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def evaluate(X_train: np.array, X_test: np.array, y_test: np.array, model_paths: list):

    results = []

    for file in model_paths:

        model = load(file)

        model_name = file[:-7]

        y_pred = model.predict(X_test)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_train)
        
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("ROC AUC", roc_auc)
        
        # Append results for summary table
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })

        return results