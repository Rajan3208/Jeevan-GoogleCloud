import logging
import concurrent.futures
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# Configure logging
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path, page_range=None):
    """Extract text directly from PDF using PyMuPDF
    
    Args:
        pdf_path (str): Path to the PDF file
        page_range (tuple): Start and end page numbers (0-indexed)
        
    Returns:
        str: Extracted text
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Determine page range
        start_page = 0
        end_page = min(len(doc), 20)  # Limit to 20 pages
        
        if page_range:
            start_page = max(0, page_range[0])
            end_page = min(len(doc), page_range[1] + 1)
        
        # Extract text from pages
        text = ""
        for page_num in range(start_page, end_page):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_with_ocr(image, fast_mode=False):
    """Extract text from image using OCR
    
    Args:
        image (PIL.Image): Image to process
        fast_mode (bool): If True, use faster but less accurate settings
        
    Returns:
        str: Extracted text
    """
    try:
        # Configure OCR settings based on mode
        if fast_mode:
            config = '--psm 1 --oem 1'  # Fast mode with automatic page segmentation
        else:
            config = '--psm 3 --oem 3'  # More accurate with fully automatic page segmentation
        
        # Perform OCR
        text = pytesseract.image_to_string(image, config=config)
        return text
    except Exception as e:
        logger.error(f"Error extracting text with OCR: {str(e)}")
        return ""

def extract_text_parallel(pdf_path, images, use_ocr=True, fast_mode=False):
    """Extract text using multiple methods in parallel
    
    Args:
        pdf_path (str): Path to the PDF file
        images (list): List of PIL images from PDF pages
        use_ocr (bool): Whether to use OCR for text extraction
        fast_mode (bool): If True, use faster but less accurate settings
        
    Returns:
        tuple: (results_dict, combined_text) Dictionary of results by method and combined text
    """
    results = {}
    
    # Extract text directly from PDF (usually faster and better when text is available)
    pdf_text = extract_text_from_pdf(pdf_path)
    results['pdf_direct'] = pdf_text
    
    # If OCR is requested and there are images, extract text with OCR
    if use_ocr and images:
        # For fast mode, process fewer images
        if fast_mode and len(images) > 5:
            # Take first, middle, and last pages for a representative sample
            sample_indices = [0, len(images)//2, len(images)-1]
            sample_images = [images[i] for i in sample_indices if i < len(images)]
        else:
            sample_images = images
        
        # Process images in parallel using thread pool
        ocr_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(extract_text_with_ocr, img, fast_mode): idx for idx, img in enumerate(sample_images)}
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    ocr_text = future.result()
                    ocr_results.append(ocr_text)
                except Exception as e:
                    logger.error(f"Error processing image {idx}: {str(e)}")
        
        # Combine OCR results
        results['ocr'] = "\n\n".join(ocr_results)
    
    # Combine all text results, prioritizing PDF direct extraction if it has content
    if pdf_text.strip() and len(pdf_text) > 100:
        # If PDF direct extraction has reasonable content, use it as primary
        combined_text = pdf_text
    elif 'ocr' in results and results['ocr'].strip():
        # Otherwise use OCR results if available
        combined_text = results['ocr']
    else:
        # Fallback to whatever
