# Project Title: E-Commerce Customer Behavior Analysis and Model Deployment

This project analyzes e-commerce customer data to build machine learning models, which are then evaluated and deployed in a production environment using `mlflow` for experiment tracking and Docker for containerization. The models predict customer behaviors, such as purchase intent, based on the features in the dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Environment](#setup-and-environment)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Deployment with MLflow](#deployment-with-mlflow)
- [Model Tracking](#model-tracking)
- [Dockerization](#dockerization)

---

## Project Overview

This project leverages machine learning to analyze and predict customer behaviors using e-commerce data. The project includes:

1. **Data Preparation and Preprocessing**: Prepares the data by encoding categorical features and splitting it into training and testing sets.
2. **Model Training**: Trains various classification models on the dataset.
3. **Evaluation**: Evaluates models based on metrics such as `accuracy`, `F1 score`, and `ROC AUC`.
4. **Deployment**: Models are versioned and tracked using `MLflow`, and Docker is used for containerization to allow easy deployment.

## Setup and Environment

### 1. Prerequisites

- Python 3.9
- Conda

### 2. Creating a Conda Environment

To get started, create and activate the project's Conda environment:

```bash
conda create -n ecommerce-prediction python=3.9
conda activate ecommerce-prediction
```
### 3. Install Required Libraries

Install the libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data Preparation

The data is loaded from an Excel file, `data/E Commerce Dataset.xlsx`, and goes through several preparation steps:

1. **Encoding**: Categorical columns are one-hot encoded to make them suitable for machine learning models.
2. **Splitting**: The data is split into training and test sets to allow for evaluation on unseen data.
3. **Feature Scaling**: Features are scaled to standardize the data, making model training more efficient.

Refer to `data_preparation.py` and `preprocessing.py` for the code.

## Training and Evaluation

### 1. Training

The `train.py` file trains several machine learning models, including Logistic Regression, SVM, Decision Trees, and Gradient Boosting Classifiers. Each model is saved to the `models/` directory after training for future evaluation.

### 2. Evaluation

The `evaluate.py` script loads each trained model, makes predictions on the test set, and logs the following metrics with MLflow:

- **Accuracy**: Measures the proportion of correctly classified instances.
- **F1 Score**: Balances precision and recall.
- **ROC AUC**: Evaluates the model's ability to distinguish between classes.

Metrics are aggregated in a results summary for easy comparison.

## Deployment with MLflow

MLflow manages the entire machine learning lifecycle, including:

1. **Experiment Tracking**: Logs parameters, metrics, and artifacts for each model training session.
2. **Model Registry**: Tracks different versions of models.
3. **Model Serving**: Allows easy deployment and management of models.

To start the MLflow tracking UI, run:

```bash
mlflow ui
```

Access the UI at http://127.0.0.1:5000 to monitor experiment logs and manage models.

## Model Tracking

To keep track of model versions and create new ones only when models are updated, MLflow’s model registry and experiment tracking functionalities are used. Code to run the full pipeline is in `main.py`.

## Dockerization

This project uses Docker to ensure reproducibility and consistent deployment.

1. **Dockerfile**  
   The Dockerfile builds a container with all required dependencies.

2. **Building and Running the Docker Container**  
   To build and run the Docker container, use the following commands:

   ```bash
   docker build -t ecommerce-prediction .
   docker run -p 5000:5000 ecommerce-prediction
    ```
    This will start an MLflow tracking server in a container to track and manage experiments.