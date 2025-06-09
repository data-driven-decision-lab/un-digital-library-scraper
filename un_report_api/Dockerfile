# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

# To copy requirements.txt from the project root, we need to go one level up.
# This assumes requirements.txt is in the parent directory of the build context.
COPY ../requirements.txt /code/requirements.txt

# Install build dependencies, install packages, then remove dependencies
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy the 'app' directory (which is directly inside the un_report_api/ context)
# into /code/app within the image.
COPY ./app /code/app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/code/app"]