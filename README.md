# MNIST Digit Classifier

Project created as part of my interview preparation for the 'ml.institute' 6-week programme starting on the 31st of March 2025.

## Project Overview

I will approach the project by separating it into different components and combining code on each step:

1. **PyTorch Model**: building a convolutional neural network trained on the MNIST dataset to recognise handwritten digits
2. **Interactive Web Interface**: create Streamlit application allowing users to draw digits and get predictions
3. **Database Logging**: store log predictions and user feedback in a PostgreSQL database
4. **Containerisation**: setup Docker for all components
5. **Deployment**: create a step-by-step instructions for server setup and I will try to deploy on a self-managed server

## Possible Project Structure

```
├── app/                    # Streamlit web application
│   ├── app.py              # Main Streamlit application
│   └── utils.py            # Utility functions for the app
├── database/               # Database setup and utilities
│   ├── init.sql            # SQL initialisation script
│   └── db_utils.py         # Database utility functions
├── docker/                 # Docker configuration
│   ├── Dockerfile.model    # Dockerfile for the model service
│   ├── Dockerfile.app      # Dockerfile for the Streamlit app
│   └── Dockerfile.db       # Dockerfile for PostgreSQL
├── model/                  # PyTorch model training and inference
│   ├── train.py            # Script to train the model
│   ├── model.py            # Model architecture definition
│   └── inference.py        # Inference utilities
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup and Installation

### Local Development and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ml.institute-MNISTclassifier.git
   cd ml.institute-MNISTclassifier
   cd MNIST-project
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python model/train.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```


