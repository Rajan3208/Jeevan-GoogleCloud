#!/bin/bash

echo "Starting up service..."
echo "Creating necessary directories..."
mkdir -p /app/models/transformers_cache

# Check if we should skip model loading on startup
if [ "$SKIP_MODELS_ON_STARTUP" = "true" ]; then
  echo "Skipping model loading on startup (will load on-demand)"
else
  echo "Loading lightweight models..."
  python -c "from utils.insight_generator import load_models; load_models(preload_all=False)" || echo "Model loading failed but continuing startup"
fi

echo "Starting server with Gunicorn..."
exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 300 app:app
