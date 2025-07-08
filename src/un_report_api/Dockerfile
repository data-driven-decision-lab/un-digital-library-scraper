# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy only the processed and reference data directories needed by the API.
COPY ../../data/processed /app/data/processed
COPY ../../data/reference /app/data/reference

# Copy the local requirements.txt
COPY requirements.txt /app/requirements.txt

# Install build dependencies, install packages, then remove dependencies
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt && \
    apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy the 'app' directory
COPY ./app /app/app

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]