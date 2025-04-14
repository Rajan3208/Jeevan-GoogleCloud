import os
import time
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging

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
    model_status = load_models()
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
        
        if file_ext == 'pdf':
            # Process PDF
            images = convert_pdf_to_images(file_path)
            text_results, combined_text = extract_text_parallel(file_path, images, use_ocr)
        else:
            # Process single image
            from PIL import Image
            import pytesseract
            
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            text_results = {'pytesseract': text}
            combined_text = text
        
        # Generate insights
        insights_details, insights_summary = generate_insights(combined_text, fast_mode)
        
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
        # Generate insights
        insights_details, insights_summary = generate_insights(text, fast_mode)
        
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
    # Preload models at startup
    logger.info("Preloading models...")
    load_models()
    
    # Start server
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)