import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import load
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def evaluate(X_train, X_test, y_test, model_paths):
    results = []
    
    for file in model_paths:
        model = load(file)
        model_name = file.split('/')[-1][:-7]  # Extracts model name from file path

        with mlflow.start_run(run_name=f"{model_name}"):
            # Make predictions
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Log the model with signature and example input
                signature = infer_signature(X_test, y_pred)
                print()
                mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_train)
                
                # Log metrics
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

            except:
                print()
                print('#############################')
                print('#### File is NOT A MODEL ####')
                print('#############################')
                pass
    
    return results
