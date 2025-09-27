# Use an official Python runtime as a parent image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the app directory
COPY ./app /app/app

# Set the Python path to include the app directory
ENV PYTHONPATH="/app/app:$PYTHONPATH"

# Change working directory to app directory for correct imports
WORKDIR /app/app

# Expose port 8080 (Cloud Run standard)
EXPOSE 8080

# Use uvicorn with optimized settings for Cloud Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]