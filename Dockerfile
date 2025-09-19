# Use an official Python runtime as a parent image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies for Chrome and Selenium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the entire source code
COPY src/ /app/src/

# Set the Python path to include the src directory
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create a simple entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "scraper" ]; then\n\
    echo "Running UN Scraper Pipeline..."\n\
    python -c "from un_data_pipeline.scraper_pipeline import main; main()"\n\
elif [ "$1" = "api" ]; then\n\
    echo "Running UN Report API..."\n\
    cd src/un_report_api/app\n\
    uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 --proxy-headers\n\
else\n\
    echo "Usage: docker run <image> [scraper|api]"\n\
    echo "  scraper: Run the UN voting data scraper"\n\
    echo "  api: Run the UN report API server"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port 8080 for the API
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default to running the scraper
CMD ["scraper"]
