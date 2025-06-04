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
# This is done first to leverage Docker cache for dependencies if they haven't changed
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
# --upgrade ensures latest versions of dependencies (within constraints of requirements.txt)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire 'app' directory (which includes your FastAPI app, 
# report_generator, models, country_iso_map, and the dashboard_output data)
# into the container at /code/app
COPY ./app /code/app

# Expose port 8000 to the outside world (the port Uvicorn will run on)
# This informs Docker that the container listens on this port at runtime
EXPOSE 8000

# Command to run the Uvicorn server when the container launches
# This assumes uvicorn is installed via requirements.txt
# It tells uvicorn to run the 'app' instance found in /code/app/main.py
# Using "app.main:app" ensures Python can find the main module within the app package
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 

# Assuming your Dockerfile context is the project root
# and your WORKDIR in the Docker image is something like /code
COPY ./un_report_api/app/required_csvs /code/un_report_api/app/required_csvs