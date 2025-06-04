    # Use an official Python runtime as a parent image
    # Using python 3.9 as an example, you can choose 3.10, 3.11 etc.
    # -slim variants are smaller and good for production
    FROM python:3.9-slim

    # Set environment variables to prevent Python from writing .pyc files and to buffer output
    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1

    # Set the working directory in the container
    WORKDIR /code

    # Copy the requirements file into the container at /code
    COPY ./requirements.txt /code/requirements.txt

    # Install any needed packages specified in requirements.txt
    RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

    # Copy your entire application structure from un_report_api/app
    # into /code/app within the image.
    COPY ./un_report_api/app /code/app

    # Expose port 8000
    EXPOSE 8000

    # Command to run the Uvicorn server when the container launches
    # Using --app-dir to ensure Uvicorn finds main:app correctly within /code/app.
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/code/app"]