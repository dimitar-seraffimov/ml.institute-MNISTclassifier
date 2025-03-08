FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/saved_models /app/.streamlit

# Create Streamlit config file
RUN echo '[server]\nport = 8080\naddress = "0.0.0.0"\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n\n[global]\ndevelopmentMode = false' > /app/.streamlit/config.toml

# Copy the application code
COPY . .

# Train a new model during the build process
RUN echo "Training a new MNIST model..." && \
    python run_model.py --mode train --epochs 10 --model-path /app/saved_models/mnist_classifier.pth

# Verify if model file exists (for debugging)
RUN ls -la /app/saved_models/ || echo "saved_models directory is empty"
RUN python -c "import os; print('Model file exists:', os.path.exists('/app/saved_models/mnist_classifier.pth')); print('Model file size:', os.path.getsize('/app/saved_models/mnist_classifier.pth') if os.path.exists('/app/saved_models/mnist_classifier.pth') else 'N/A')"

# Expose the port that will be used by Cloud Run
EXPOSE 8080

# Set environment variables with defaults that can be overridden at runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/saved_models/mnist_classifier.pth \
    DB_HOST="34.142.21.157" \
    DB_PORT=5432 \
    DB_NAME=mnist_db \
    DB_USER=postgres \
    DB_PASSWORD=master \
    DB_INSTANCE_CONNECTION_NAME="mnist-deploy-452915:europe-west2:localhost" \
    PORT=8080

# Command to run the application directly with Streamlit
CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8080", "--server.address=0.0.0.0"] 