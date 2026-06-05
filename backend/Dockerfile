# Use a lightweight Python base image with better library support
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create weights directory for volume mount
RUN mkdir -p /app/weights

# Expose the Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"]
