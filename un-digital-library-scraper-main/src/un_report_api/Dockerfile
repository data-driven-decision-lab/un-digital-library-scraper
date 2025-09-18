# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy the local requirements.txt
COPY requirements.txt /app/requirements.txt

# Install build dependencies, install packages, then remove dependencies
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt && \
    apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy the 'app' directory (which now contains the data files)
COPY ./app /app/app

# Set the Python path to include the app directory
ENV PYTHONPATH="/app/app:$PYTHONPATH"

# Change working directory to app directory for correct imports
WORKDIR /app/app

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]