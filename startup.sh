#!/bin/bash
echo "Starting up service..."
echo "Creating necessary directories..."
# No need to create this directory as it's already created in the Dockerfile
# mkdir -p /models/transformers_cache 

# Check if we should skip model loading on startup
if [ "$SKIP_MODELS_ON_STARTUP" = "true" ]; then
  echo "Skipping model loading on startup (will load on-demand)"
else
  echo "Loading lightweight models..."
  python -c "from utils.insight_generator import load_models; load_models(preload_all=False)" || echo "Model loading failed but continuing startup"
fi

# Use the PORT environment variable instead of hardcoding 8080
echo "Starting server with Gunicorn on port $PORT..."
exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 app:app
