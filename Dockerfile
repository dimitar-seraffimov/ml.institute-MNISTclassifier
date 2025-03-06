FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for saved models and streamlit config
RUN mkdir -p saved_models .streamlit

# Create Streamlit config file
RUN echo '[server]\nport = 8080\naddress = "0.0.0.0"\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n\n[global]\ndevelopmentMode = false' > .streamlit/config.toml

# Copy the rest of the application
COPY . .

# Create a dummy model file if none exists (for testing only)
RUN if [ ! -f saved_models/mnist_classifier.pth ] && [ ! -f model/mnist_classifier.pth ]; then \
    echo "Creating dummy model file for testing" && \
    python -c "import torch; torch.save({'state_dict': {}}, 'saved_models/mnist_classifier.pth')" || echo "Failed to create dummy model"; \
    fi

# Verify if model file exists (for debugging)
RUN ls -la saved_models/ || echo "saved_models directory is empty"
RUN ls -la model/ || echo "model directory is empty"

# Expose the port that will be used by Cloud Run
# Cloud Run will set the PORT environment variable
EXPOSE 8080

# Set environment variables with defaults that can be overridden at runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/saved_models/mnist_classifier.pth \
    DB_HOST=localhost \
    DB_PORT=5432 \
    DB_NAME=mnist_db \
    DB_USER=postgres \
    DB_PASSWORD=postgres \
    DATABASE_URL="" \
    PORT=8080

# Command to run the application using run_streamlit.py
CMD ["python", "run_streamlit.py"] 