# Use an official Python runtime as a parent image
# Choose a version compatible with your project (e.g., 3.11)
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disc (standard in containers)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to terminal without being buffered
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     # Add any system packages required by your dependencies here
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt requirements.txt
# Consider using a requirements-prod.txt if you have separate dev dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This includes your src directory, models, notebooks (if needed), etc.
COPY ./src ./src
# If your models are not in src, copy them separately
# COPY ./models ./models
# Ensure the model directory structure matches what's expected by src/api/main.py
# (e.g., if main.py expects models/model.pkl, ensure models dir is copied to /app/models)

# Make port 8000 available to the world outside this container
# This should match the port Uvicorn runs on
EXPOSE 8000

# Define environment variables for the application
# These can be overridden at runtime (e.g., docker run -e MODEL_DIR=/path/to/models ...)
# Set defaults that work within the container structure
ENV MODEL_DIR=/app/models
# ENV MODEL_FILENAME=your_model.pkl # Override if needed
# ENV W2V_MODEL_FILENAME=your_w2v.model # Override if needed

# --- Healthcheck --- 
# Optional: Add a healthcheck to verify the API is responsive
# HEALTHCHECK --interval=15s --timeout=5s --start-period=30s \
#   CMD curl --fail http://localhost:8000/ || exit 1

# Run the application using Uvicorn
# The command should point to the FastAPI app instance within your module
# Ensure the host is 0.0.0.0 to accept connections from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 