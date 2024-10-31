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
                registered_model = mlflow.register_model(model_uri, model_name)
            except mlflow.exceptions.RestException:
                # Model already registered; MLflow will handle new versions automatically
                pass
