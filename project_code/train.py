import mlflow
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def train(X_train, y_train):
    """
    Train and register multiple classification models using MLflow.

    This function trains a variety of classification models on the given training data,
    logs their parameters to MLflow, saves each model as a file, logs the model file
    as an artifact in MLflow, and registers the model in the MLflow Model Registry.
    If the model already exists in the registry, a new version is created.

    Args:
        X_train (numpy.ndarray or pandas.DataFrame): The training data features used for model training.
        y_train (numpy.ndarray or pandas.Series): The target labels for the training data.

    Models:
        The function trains the following models:
        - Logistic Regression
        - Support Vector Machine (SVM)
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - K-Nearest Neighbors (KNN)
        - Naive Bayes
        - XGBoost

    Returns:
        None

    Raises:
        mlflow.exceptions.RestException: If there is an issue with model registration in MLflow,
        except when the model already exists (in which case a new version is created automatically).

    Example:
        >>> X_train, y_train = load_training_data()
        >>> train(X_train, y_train)

    Notes:
        - Model parameters are logged for each model that supports `get_params`.
        - Models are saved in the `models` folder, with each model file named after its respective
          model type (e.g., 'Logistic Regression.joblib').
        - Models are registered in MLflow under their respective names, and if a model with the
          same name exists, MLflow creates a new version.
    """

    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier()
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # Train the model
            model.fit(X_train, y_train)

            # Save model
            model_path = f"models/{model_name}.joblib"
            dump(model, model_path)
            # Log model path as artifact
            mlflow.log_artifact(model_path, artifact_path="models")

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"

            # Register the model, and if it exists, create a new version
            try:
                mlflow.register_model(model_uri, model_name)
            except mlflow.exceptions.RestException:
                # Model already registered; MLflow will handle new versions automatically
                pass
