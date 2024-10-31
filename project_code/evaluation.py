import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import load
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def evaluate(X_train, X_test, y_test, model_paths):
    """
    Evaluate multiple models on test data and log the results to MLflow.

    This function loads models from the specified paths, makes predictions on the test data,
    calculates evaluation metrics (accuracy, F1 score, and ROC AUC), logs these metrics and
    the model to MLflow, and returns a summary of the results for each model.

    Args:
        X_train (numpy.ndarray or pandas.DataFrame): The training data features.
            Used as an input example for logging the model to MLflow.
        X_test (numpy.ndarray or pandas.DataFrame): The test data features on which predictions
            are made by the models.
        y_test (numpy.ndarray or pandas.Series): The true labels for the test data.
        model_paths (list of str): A list of file paths to the serialized model files to be evaluated.

    Returns:
        list of dict: A list containing a dictionary for each model with keys `Model`, `Accuracy`,
        `F1 Score`, and `ROC AUC`, representing the model name and its corresponding performance metrics.

    Raises:
        FileNotFoundError: If any model file in `model_paths` cannot be found.
        Exception: If there is an issue with loading or evaluating a model.

    Example:
        >>> X_train, X_test, y_test = load_data()
        >>> model_paths = ["models/model1.joblib", "models/model2.joblib"]
        >>> results = evaluate(X_train, X_test, y_test, model_paths)
        >>> print(results)
        [{'Model': 'model1', 'Accuracy': 0.85, 'F1 Score': 0.84, 'ROC AUC': 0.88},
         {'Model': 'model2', 'Accuracy': 0.82, 'F1 Score': 0.81, 'ROC AUC': 0.85}]

    Notes:
        - If a model does not support probability prediction, the ROC AUC score will be calculated
          using zero probabilities.
        - Non-model files in `model_paths` are skipped with a warning message printed.
    """

    results = []

    for file in model_paths:
        model = load(file)
        model_name = file.split('/')[-1][:-7]  # Extracts model name from file path

        with mlflow.start_run(run_name=f"{model_name}"):
            # Make predictions
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(
                    X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))

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
