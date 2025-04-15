import os
import time
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging
import concurrent.futures
from functools import lru_cache

# Import utility modules
from utils.pdf_processor import convert_pdf_to_images
from utils.text_extraction import extract_text_parallel
from utils.insight_generator import generate_insights, load_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Global thread pool for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Cache for model results to avoid redundant processing
@lru_cache(maxsize=100)
def cached_generate_insights(text_hash, fast_mode):
    # We use the hash of text as the key since text itself might be too large for a cache key
    return generate_insights.original_function(text_hash, fast_mode)

# Patch the generate_insights function with a cached version
generate_insights.original_function = generate_insights
generate_insights = lru_cache(maxsize=20)(generate_insights)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/load-models', methods=['GET'])
def preload_models():
    """Endpoint to preload models"""
    start_time = time.time()
    model_status = load_models(preload_all=True)  # Modified to preload all models
    duration = time.time() - start_time
    return jsonify({
        'status': 'Models loaded successfully',
        'model_status': model_status,
        'duration_seconds': duration
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Main endpoint to analyze PDF or image documents"""
    start_time = time.time()
    
    # Check if file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Please upload a PDF or image ({", ".join(ALLOWED_EXTENSIONS)})'}), 400
    
    # Get analysis mode
    fast_mode = request.form.get('fast_mode', 'true').lower() == 'true'
    use_ocr = request.form.get('use_ocr', 'true').lower() == 'true'
    
    logger.info(f"Processing file: {file.filename} (fast_mode: {fast_mode}, use_ocr: {use_ocr})")
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process based on file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        # Use a lower sampling rate for large PDFs in fast mode
        sample_rate = 2 if fast_mode else 1  # Sample every other page in fast mode
        
        if file_ext == 'pdf':
            # Process PDF with optimized settings
            max_pages = 10 if fast_mode else 20  # Limit pages in fast mode
            images = convert_pdf_to_images(file_path, max_pages=max_pages, sample_rate=sample_rate)
            
            # Process text extraction in parallel
            text_results, combined_text = extract_text_parallel(
                file_path, 
                images, 
                use_ocr=use_ocr,
                fast_mode=fast_mode
            )
        else:
            # Process single image with optimized settings
            from PIL import Image
            import pytesseract
            
            img = Image.open(file_path)
            
            # Resize large images for faster processing in fast mode
            if fast_mode and (img.width > 2000 or img.height > 2000):
                scale_factor = 0.5
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Use simpler OCR config in fast mode
            config = '--psm 1 --oem 1' if fast_mode else '--psm 3 --oem 3'
            text = pytesseract.image_to_string(img, config=config)
            text_results = {'pytesseract': text}
            combined_text = text
        
        # Submit insight generation to thread pool to avoid blocking
        if fast_mode:
            # For fast mode, use a simpler approach directly
            future = executor.submit(generate_insights, combined_text[:10000], fast_mode)  # Limit text length
        else:
            future = executor.submit(generate_insights, combined_text, fast_mode)
        
        # Get results with timeout
        timeout = 30 if fast_mode else 60
        insights_details, insights_summary = future.result(timeout=timeout)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up temp file
        os.remove(file_path)
        
        # Prepare response
        response = {
            'status': 'success',
            'processing_time_seconds': processing_time,
            'insights_summary': insights_summary,
            'insights_details': insights_details,
            'text_extraction': {
                'methods_used': list(text_results.keys()),
                'text_length': len(combined_text),
                'text_sample': combined_text[:500] + '...' if len(combined_text) > 500 else combined_text
            }
        }
        
        return jsonify(response), 200
    
    except concurrent.futures.TimeoutError:
        logger.error("Processing timed out")
        return jsonify({
            'error': 'Processing timed out. Try using fast_mode=true for quicker results.',
            'status': 'timeout'
        }), 408
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Error processing document: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Endpoint to analyze raw text"""
    start_time = time.time()
    
    # Get text content from request
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    fast_mode = data.get('fast_mode', True)
    
    try:
        # Limit text length for fast processing
        if fast_mode and len(text) > 10000:
            text = text[:10000]
            
        # Generate insights with timeout
        future = executor.submit(generate_insights, text, fast_mode)
        timeout = 30 if fast_mode else 60
        insights_details, insights_summary = future.result(timeout=timeout)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'status': 'success',
            'processing_time_seconds': processing_time,
            'insights_summary': insights_summary,
            'insights_details': insights_details,
            'text_length': len(text)
        }
        
        return jsonify(response), 200
    
    except concurrent.futures.TimeoutError:
        logger.error("Text analysis timed out")
        return jsonify({
            'error': 'Processing timed out. Try using fast_mode=true for quicker results.',
            'status': 'timeout'
        }), 408
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Error analyzing text: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'name': 'PDF Analysis API',
        'version': '1.0.0',
        'endpoints': {
            '/analyze': 'POST - Analyze PDF or image documents',
            '/analyze-text': 'POST - Analyze raw text',
            '/health': 'GET - Health check',
            '/load-models': 'GET - Preload models'
        },
        'documentation': 'See README.md for detailed usage instructions'
    }), 200

# Global variable to track model loading status
models_loaded = False

@app.before_first_request
def load_models_on_startup():
    """Load lightweight models on startup"""
    global models_loaded
    if not models_loaded:
        logger.info("Loading lightweight models on startup...")
        load_models(preload_all=False)  # Only load essential models
        models_loaded = True
        logger.info("Essential models loaded successfully")

if __name__ == '__main__':
    # Start server
    port = int(os.environ.get('PORT', 8080))
    
    # Preload models in a separate thread to avoid blocking startup
    import threading
    threading.Thread(target=load_models_on_startup).start()
    
    app.run(host='0.0.0.0', port=port)
