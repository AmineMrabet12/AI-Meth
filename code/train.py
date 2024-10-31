from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from joblib import dump
import mlflow
import mlflow.sklearn


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

    # experiment_name = "AI-Meth 1st Xp"
    # mlflow.set_experiment(experiment_name)

    # mlflow.set_tracking_uri("mlruns")

    # results = []

    # Train each model, make predictions, and evaluate performance
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Log model parameters if applicable
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())
            
            # Train model
            model.fit(X_train, y_train)

            # Save the model to a file and log it as an artifact
            model_path = f"models/{model_name}.joblib"

            dump(model, model_path)

    return models
            # mlflow.log_artifact(model_path, artifact_path="models")
            # mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_train)

            
            
            