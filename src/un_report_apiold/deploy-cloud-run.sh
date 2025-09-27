#!/bin/bash

# Google Cloud Run Deployment Script for UN Report API
# Make sure you have gcloud CLI installed and authenticated

set -e

# Configuration
PROJECT_ID="your-project-id"  # Replace with your Google Cloud Project ID
SERVICE_NAME="un-report-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying UN Report API to Google Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region: $REGION"
echo "Image: $IMAGE_NAME"

# Step 1: Build the Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME .

# Step 2: Push to Google Container Registry
echo "⬆️ Pushing image to Google Container Registry..."
docker push $IMAGE_NAME

# Step 3: Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 0 \
  --timeout 300 \
  --concurrency 80 \
  --set-env-vars "SUPABASE_URL=https://gjakiqtayqltssvbzasd.supabase.co" \
  --set-env-vars "SUPABASE_KEY=your-supabase-service-role-key"

echo "✅ Deployment complete!"
echo "🌐 Your API is now available at:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"

echo ""
echo "📋 Next steps:"
echo "1. Update the SUPABASE_KEY environment variable with your actual service role key"
echo "2. Test the API endpoints"
echo "3. Set up monitoring and logging"
echo "4. Configure custom domain if needed"
