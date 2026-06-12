# Use a lightweight Python base image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies (ONNX Runtime is light!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create weights directory
RUN mkdir -p /app/weights

# Expose the Flask port
EXPOSE 4000

# Set environment variables (MODEL_URL is optional - for S3 download)
ENV PORT=4000
ENV DEBUG=false
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"]
