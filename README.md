# PDF Analysis API

A fast and comprehensive PDF and image analysis API that extracts text and generates insights from documents.

## Features

- Fast document analysis (results in seconds)
- Multiple text extraction methods (PyPDF2, Tesseract OCR, EasyOCR, Langchain)
- Comprehensive document insights:
  - Named entity recognition
  - Topic modeling
  - Sentiment analysis
  - Text summarization
- Optimized for Google Cloud deployment
- Docker containerized

## API Endpoints

### `/analyze` (POST)

Analyze PDF documents or images.

**Parameters:**
- `file`: The PDF or image file to analyze (required)
- `fast_mode`: Enable faster processing (default: true)
- `use_ocr`: Enable OCR for better text extraction (default: true)

**Example Response:**
```json
{
  "status": "success",
  "processing_time_seconds": 2.5,
  "insights_summary": "Document contains 25 named entities including John Smith, New York, 2023. Key topics focus on financial reporting, quarterly results, market analysis.",
  "insights_details": {
    "spacy": {
      "entities": {"John Smith": "PERSON", "New York": "GPE"},
      "key_phrases": ["financial report", "quarterly results"]
    },
    "topics": [["financial", "report", "quarterly"], ["market", "analysis", "stock"]]
  },
  "text_extraction": {
    "methods_used": ["pypdf", "pytesseract"],
    "text_length": 3500,
    "text_sample": "This financial report covers the third quarter of 2023..."
  }
}
```

### `/analyze-text` (POST)

Analyze raw text directly.

**Parameters:**
- `text`: The text to analyze (required)
- `fast_mode`: Enable faster processing (default: true)

### `/health` (GET)

Health check endpoint.

### `/load-models` (GET)

Preload models to make subsequent requests faster.

## Deployment on Google Cloud

### Local Testing

1. Build the Docker image:
   ```
   docker build -t pdf-analysis-api .
   ```

2. Run the container locally:
   ```
   docker run -p 8080:8080 pdf-analysis-api
   ```

3. Test the API:
   ```
   curl -X GET http://localhost:8080/health
   ```

### Deploy to Google Cloud Run

1. Tag your Docker image for Google Container Registry:
   ```
   docker tag pdf-analysis-api gcr.io/[PROJECT-ID]/pdf-analysis-api
   ```

2. Push the image:
   ```
   docker push gcr.io/[PROJECT-ID]/pdf-analysis-api
   ```

3. Deploy to Cloud Run
