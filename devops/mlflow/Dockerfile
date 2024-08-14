# Base Image
FROM python:3.8-slim

# Set Maintainer Label
LABEL maintainer="Kailash Lohar <kailashlohariitkgp@gmail.com>"

# Install MLflow and dependencies
RUN pip install mlflow psycopg2-binary

# Set Working Directory
WORKDIR /mlflow

# Expose Port for MLflow
EXPOSE 5001

# Set Default Command for starting MLflow
CMD ["mlflow", "server", "--backend-store-uri", "postgresql://kailash:ml_model@postgres_db:5432/mlflow_db", "--default-artifact-root", "/mlflow/mlruns", "--host=0.0.0.0", "--port=5001"]
