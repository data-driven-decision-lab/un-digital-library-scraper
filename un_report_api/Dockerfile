# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

# Copy the local requirements.txt
COPY requirements.txt /code/requirements.txt

# Install build dependencies, install packages, then remove dependencies
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy the 'app' directory
COPY ./app /code/app

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]