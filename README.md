# E-Commerce Customer Behavior Analysis and Model Deployment

This project analyzes e-commerce customer data to build machine learning models, which are then evaluated and deployed in a production environment using `mlflow` for experiment tracking and Docker for containerization. The models predict customer behaviors, such as purchase intent, based on the features in the dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Environment](#setup-and-environment)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Deployment with MLflow](#deployment-with-mlflow)
- [Dockerization](#dockerization)
- [Documentation with Sphinx](#documentation-with-sphinx)
- [model interpretability with shap](#model-interpretability-with-shap)

---

## Project Overview

This project leverages machine learning techniques to analyze and predict customer behaviors using e-commerce data. The primary goals include understanding customer patterns and predicting future purchasing decisions based on various features within the dataset. The project encompasses the following key components:

1. **Data Preparation and Preprocessing**: 
   - Cleans and prepares the data by handling missing values, encoding categorical features, and splitting the dataset into training and testing sets to ensure effective model training and evaluation.

2. **Model Training**: 
   - Trains a variety of classification models, including Logistic Regression, Support Vector Machines (SVM), Decision Trees, and Gradient Boosting Classifiers, to predict customer behaviors based on the prepared dataset.

3. **Evaluation**: 
   - Assesses the performance of each trained model using key metrics such as accuracy, F1 score, and ROC AUC, allowing for an informed comparison of model effectiveness.

4. **Interpretability**:
   - Utilizes **SHAP** (SHapley Additive exPlanations) to explain the output of machine learning models, providing insights into feature importance and how different features impact predictions. This helps in understanding the decision-making process of the models.

5. **Documentation**:
   - Implements **Sphinx** for comprehensive project documentation, making it easier for users to understand the project structure, usage, and functionalities. Sphinx allows for easy generation of documentation from the codebase, ensuring that the project is well-documented and maintainable.

6. **Tracking**: 
   - Utilizes **MLflow** for versioning and tracking model parameters, metrics, and artifacts throughout the machine learning lifecycle, ensuring reproducibility and transparency in experiments.

7. **Deployment**: 
   - Implements **Docker** for containerization, facilitating easy deployment and management of the machine learning models in production environments.


This comprehensive approach ensures that the project is not only focused on building models but also on maintaining a robust and scalable machine learning workflow, with an emphasis on interpretability and documentation.

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
Run This Command to install MLflow

```bash
pip install mlflow
```

MLflow manages the entire machine learning lifecycle, including:

1. **Experiment Tracking**: Logs parameters, metrics, and artifacts for each model training session.
2. **Model Registry**: Tracks different versions of models.
3. **Model Serving**: Allows easy deployment and management of models.

To start the MLflow tracking UI, run:

```bash
mlflow ui
```

Access the UI at http://127.0.0.1:5000 to monitor experiment logs and manage models.

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

## Documentation with Sphinx

This project utilizes Sphinx for generating documentation. To build the documentation, navigate to the docs/ directory and run:

### 1. Install Sphinx

To get started with Sphinx, you need to install it first. You can do this using pip:

```bash
pip install sphinx
```

### 2.  Initialize Sphinx
Once Sphinx is installed, navigate to your project directory and run the following command to initialize Sphinx:

```bash
sphinx-quickstart
```
This command will prompt you with a series of questions to configure your Sphinx project.

After answering the prompts, Sphinx will create the necessary directories and files in a new `docs` folder.

### 3. Build Documentation
To generate HTML documentation, run the following command from the `docs` directory:
```bash
make html
```
This command creates HTML documentation in the `docs/build/html` directory.


## Model Interpretability with SHAP

To enhance model interpretability, this project incorporates SHAP (SHapley Additive exPlanations). SHAP values help understand the impact of each feature on the model's predictions, providing insights into model behavior.

Example output:
<div align="center">
<h2>WaterFall plot</h2>
    <img src="SHAP output/WaterFall.png" alt="Description of Image" />
</div>
<div align="center">
<h2>Beeswarm plot</h2>
    <img src="SHAP output/Beeswarm plot.png" alt="Description of Image" />
</div>
<div align="center">
<h2>Dependence plot</h2>
    <img src="SHAP output/Dependence plot.png" alt="Description of Image" />
</div>
<div align="center">
<h2>Mean SHAP plot</h2>
    <img src="SHAP output/Mean SHAP plot.png" alt="Description of Image" />
</div>