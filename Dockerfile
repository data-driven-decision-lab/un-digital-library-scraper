# Use an official Python runtime as a parent image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy only the API source code
COPY src/un_report_api/ /app/src/un_report_api/

# Set the Python path to include the src directory
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create a simple entrypoint script for API only
RUN echo '#!/bin/bash\n\
echo "Running UN Report API..."\n\
cd src/un_report_api/app\n\
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 --proxy-headers' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port 8080 for the API
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
