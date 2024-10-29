# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all necessary directories and files into the container
COPY code/ /app/code/
COPY data/ /app/data/
COPY mlruns/ /app/mlruns/
COPY models/ /app/models/
COPY notebooks/ /app/notebooks/

# Optional: Set an environment variable for MLflow if needed
# ENV MLFLOW_TRACKING_URI sqlite:///mlruns/mlflow.db

# Expose ports if your application or MLflow will use any, e.g., port 5000 for MLflow
EXPOSE 5000

# Optional: Set the default command if you have a main script, e.g., `main.py` in `code/`
# CMD ["python", "code/main.py"]
