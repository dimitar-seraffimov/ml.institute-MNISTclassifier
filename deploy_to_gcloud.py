#!/usr/bin/env python
"""
Deployment script for the MNIST Classifier application to Google Cloud Run.
This script automates the process of:
1. Setting up a Google Cloud project
2. Creating a Cloud SQL PostgreSQL instance
3. Building and pushing the Docker image to Google Container Registry
4. Deploying the application to Google Cloud Run
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

def run_command(command, description=None, check=True):
    """Run a shell command and print its output."""
    if description:
        print(f"\n{description}...")
    
    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True,
            shell=True
        )
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        if e.stderr:
            print(e.stderr)
        return False, e.stderr

def setup_gcloud(project_id, region):
    """Set up Google Cloud project and enable required APIs."""
    # Check if gcloud is installed
    success, _ = run_command("gcloud --version", "Checking if Google Cloud SDK is installed")
    if not success:
        print("❌ Google Cloud SDK is not installed or not in PATH")
        print("Please install it from https://cloud.google.com/sdk/docs/install")
        return False
    
    # Initialize gcloud and create project if it doesn't exist
    run_command(f"gcloud projects describe {project_id}", "Checking if project exists", check=False)
    run_command(f"gcloud projects create {project_id} --name='MNIST Classifier'", 
                "Creating Google Cloud project", check=False)
    
    # Set the project
    success, _ = run_command(f"gcloud config set project {project_id}", "Setting active project")
    if not success:
        return False
    
    # Enable required APIs
    apis = [
        "cloudbuild.googleapis.com",
        "run.googleapis.com",
        "sqladmin.googleapis.com",
        "artifactregistry.googleapis.com"
    ]
    
    for api in apis:
        run_command(f"gcloud services enable {api}", f"Enabling {api}")
    
    # Set the region
    run_command(f"gcloud config set run/region {region}", "Setting region")
    
    return True

def setup_cloud_sql(project_id, instance_name, region, db_name, db_user, db_password):
    """Set up a Cloud SQL PostgreSQL instance."""
    # Create the instance if it doesn't exist
    run_command(
        f"gcloud sql instances describe {instance_name}",
        "Checking if Cloud SQL instance exists",
        check=False
    )
    
    # Create the instance if it doesn't exist
    success, output = run_command(
        f"gcloud sql instances create {instance_name} "
        f"--database-version=POSTGRES_14 "
        f"--tier=db-f1-micro "
        f"--region={region} "
        f"--root-password={db_password} "
        f"--storage-size=10GB",
        "Creating Cloud SQL instance (this may take a few minutes)",
        check=False
    )
    
    # Create the database if it doesn't exist
    run_command(
        f"gcloud sql databases create {db_name} --instance={instance_name}",
        f"Creating database {db_name}",
        check=False
    )
    
    # Create the user if it doesn't exist
    run_command(
        f"gcloud sql users create {db_user} --instance={instance_name} --password={db_password}",
        f"Creating database user {db_user}",
        check=False
    )
    
    # Get the instance connection name
    success, output = run_command(
        f"gcloud sql instances describe {instance_name} --format='value(connectionName)'",
        "Getting instance connection name"
    )
    
    if not success:
        return False, None
    
    connection_name = output.strip()
    
    # Get the instance IP address
    success, output = run_command(
        f"gcloud sql instances describe {instance_name} --format='value(ipAddresses[0].ipAddress)'",
        "Getting instance IP address"
    )
    
    if not success:
        return False, None
    
    ip_address = output.strip()
    
    return True, (connection_name, ip_address)

def build_and_push_image(project_id, image_name):
    """Build and push the Docker image to Google Container Registry."""
    # Build the image
    success, _ = run_command(
        f"docker build -t gcr.io/{project_id}/{image_name} .",
        "Building Docker image"
    )
    if not success:
        return False
    
    # Configure Docker to use gcloud credentials
    run_command(
        "gcloud auth configure-docker",
        "Configuring Docker to use gcloud credentials"
    )
    
    # Push the image
    success, _ = run_command(
        f"docker push gcr.io/{project_id}/{image_name}",
        "Pushing Docker image to Google Container Registry"
    )
    
    return success

def deploy_to_cloud_run(project_id, service_name, image_name, db_connection_name, db_ip, db_name, db_user, db_password, region):
    """Deploy the application to Google Cloud Run."""
    # Deploy to Cloud Run
    success, _ = run_command(
        f"gcloud run deploy {service_name} "
        f"--image gcr.io/{project_id}/{image_name} "
        f"--platform managed "
        f"--allow-unauthenticated "
        f"--region={region} "
        f"--set-env-vars=POSTGRES_HOST={db_ip},"
        f"POSTGRES_PORT=5432,"
        f"POSTGRES_DB={db_name},"
        f"POSTGRES_USER={db_user},"
        f"POSTGRES_PASSWORD={db_password},"
        f"MODEL_PATH=/app/saved_models/mnist_classifier.pth",
        "Deploying to Cloud Run"
    )
    
    if not success:
        return False
    
    # Connect Cloud Run to Cloud SQL
    success, _ = run_command(
        f"gcloud run services update {service_name} "
        f"--add-cloudsql-instances={db_connection_name} "
        f"--region={region}",
        "Connecting Cloud Run to Cloud SQL"
    )
    
    if not success:
        return False
    
    # Get the service URL
    success, output = run_command(
        f"gcloud run services describe {service_name} "
        f"--platform managed "
        f"--region={region} "
        f"--format='value(status.url)'",
        "Getting service URL"
    )
    
    if not success:
        return False
    
    service_url = output.strip()
    
    return True, service_url

def main():
    """Main function to deploy the application to Google Cloud Run."""
    parser = argparse.ArgumentParser(description="Deploy MNIST Classifier to Google Cloud Run")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud region")
    parser.add_argument("--db-instance", default="mnist-db", help="Cloud SQL instance name")
    parser.add_argument("--db-name", default="mnist_db", help="Database name")
    parser.add_argument("--db-user", default="postgres", help="Database user")
    parser.add_argument("--db-password", default="master", help="Database password")
    parser.add_argument("--service-name", default="mnist-classifier", help="Cloud Run service name")
    parser.add_argument("--image-name", default="mnist-app", help="Docker image name")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("MNIST Classifier Deployment to Google Cloud Run")
    print("=" * 50)
    
    # Set up Google Cloud project
    if not setup_gcloud(args.project_id, args.region):
        print("❌ Failed to set up Google Cloud project")
        return False
    
    # Set up Cloud SQL
    success, sql_info = setup_cloud_sql(
        args.project_id, 
        args.db_instance, 
        args.region, 
        args.db_name, 
        args.db_user, 
        args.db_password
    )
    
    if not success:
        print("❌ Failed to set up Cloud SQL")
        return False
    
    db_connection_name, db_ip = sql_info
    
    # Build and push Docker image
    if not build_and_push_image(args.project_id, args.image_name):
        print("❌ Failed to build and push Docker image")
        return False
    
    # Deploy to Cloud Run
    success, service_url = deploy_to_cloud_run(
        args.project_id,
        args.service_name,
        args.image_name,
        db_connection_name,
        db_ip,
        args.db_name,
        args.db_user,
        args.db_password,
        args.region
    )
    
    if not success:
        print("❌ Failed to deploy to Cloud Run")
        return False
    
    print("=" * 50)
    print(f"✅ Deployment successful! Your application is available at: {service_url}")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 