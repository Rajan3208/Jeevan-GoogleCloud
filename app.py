import os
import time
import tempfile
from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import logging
import concurrent.futures
from functools import lru_cache
import threading

# Import utility modules
from utils.pdf_processor import convert_pdf_to_images
from utils.text_extraction import extract_text_parallel
from utils.insight_generator import generate_insights, load_models, check_models_loaded

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Global thread pool for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Flag for tracking model loading status
is_loading_models = False
model_loading_thread = None

# Cache for model results to avoid redundant processing
@lru_cache(maxsize=100)
def cached_generate_insights(text_hash, fast_mode):
    # We use the hash of text as the key since text itself might be too large for a cache key
    return generate_insights(text_hash, fast_mode)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models_in_background():
    """Load models in a background thread"""
    global is_loading_models
    try:
        logger.info("Starting background model loading")
        is_loading_models = True
        load_models(preload_all=False)
        logger.info("Background model loading completed")
    except Exception as e:
        logger.error(f"Error in background model loading: {str(e)}", exc_info=True)
    finally:
        is_loading_models = False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint that always returns 200 OK"""
    models_status = check_models_loaded()
    
    # Start background loading of models if not already loading and not loaded
    global is_loading_models, model_loading_thread
    if (not is_loading_models and 
        not models_status.get("all_required_loaded", False) and 
        (model_loading_thread is None or not model_loading_thread.is_alive())):
        model_loading_thread = threading.Thread(target=load_models_in_background)
        model_loading_thread.daemon = True
        model_loading_thread.start()
        logger.info("Started background model loading from health check")
    
    # Always return 200 OK for Cloud Run health checks
    return jsonify({
        'status': 'healthy', 
        'models': models_status,
        'initialization': 'in progress' if is_loading_models else 'complete' if models_status.get("all_required_loaded", False) else 'pending'
    }), 200

@app.route('/load-models', methods=['GET'])
def preload_models():
    """Endpoint to preload models"""
    global is_loading_models, model_loading_thread
    
    # Check if models are already loading
    if is_loading_models:
        return jsonify({
            'status': 'Models are currently loading in background',
            'already_in_progress': True
        }), 200
    
    start_time = time.time()
    try:
        # Start background loading if not already in progress
        if model_loading_thread is None or not model_loading_thread.is_alive():
            model_loading_thread = threading.Thread(target=load_models_in_background)
            model_loading_thread.daemon = True
            model_loading_thread.start()
            logger.info("Started background model loading from load-models endpoint")
            
        model_status = check_models_loaded()
        duration = time.time() - start_time
        return jsonify({
            'status': 'Model loading initiated in background',
            'current_model_status': model_status,
            'duration_seconds': duration
        }), 200
    except Exception as e:
        logger.error(f"Error initiating model loading: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'duration_seconds': time.time() - start_time
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Main endpoint to analyze PDF or image documents"""
    # Check if models are loaded or load them on demand
    models_status = check_models_loaded()
    if not models_status.get("all_required_loaded", False):
        # Start background loading if not already in progress
        global is_loading_models, model_loading_thread
        if not is_loading_models and (model_loading_thread is None or not model_loading_thread.is_alive()):
            model_loading_thread = threading.Thread(target=load_models_in_background)
            model_loading_thread.daemon = True
            model_loading_thread.start()
            logger.info("Started background model loading from analyze endpoint")
        
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
        
        # Check again if models are loaded
        models_status = check_models_loaded()
        if not models_status.get("min_required_loaded", False):
            return jsonify({
                'status': 'processing_incomplete',
                'error': 'Models are still loading. Please try again in a few moments.',
                'text_extraction': {
                    'methods_used': list(text_results.keys()),
                    'text_length': len(combined_text),
                    'text_sample': combined_text[:500] + '...' if len(combined_text) > 500 else combined_text
                }
            }), 202
            
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
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {str(e)}")
        
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
        try:
            os.remove(file_path)
        except:
            pass
        return jsonify({
            'error': 'Processing timed out. Try using fast_mode=true for quicker results.',
            'status': 'timeout'
        }), 408
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        try:
            os.remove(file_path)
        except:
            pass
        return jsonify({
            'error': f'Error processing document: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Endpoint to analyze raw text"""
    # Check if models are loaded
    models_status = check_models_loaded()
    if not models_status.get("min_required_loaded", False):
        # Start background loading if not already in progress
        global is_loading_models, model_loading_thread
        if not is_loading_models and (model_loading_thread is None or not model_loading_thread.is_alive()):
            model_loading_thread = threading.Thread(target=load_models_in_background)
            model_loading_thread.daemon = True
            model_loading_thread.start()
            logger.info("Started background model loading from analyze-text endpoint")
            
        return jsonify({
            'status': 'processing_delayed',
            'error': 'Models are still loading. Please try again in a few moments.'
        }), 202
        
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

if __name__ == '__main__':
    # Start server
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
