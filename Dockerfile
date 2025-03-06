FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for saved models
RUN mkdir -p saved_models

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/saved_models/mnist_classifier.pth \
    DB_HOST=db \
    DB_PORT=5432 \
    DB_NAME=mnist_db \
    DB_USER=postgres \
    DB_PASSWORD=postgres

# Command to run the application
CMD ["python3", "streamlit/app.py"] 