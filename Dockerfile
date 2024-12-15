# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from current directory to working directory
COPY . .

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app