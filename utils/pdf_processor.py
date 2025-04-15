import logging
import fitz  # PyMuPDF
from PIL import Image
import io
import os

# Configure logging
logger = logging.getLogger(__name__)

def convert_pdf_to_images(pdf_path, max_pages=20, sample_rate=1, dpi=200):
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path (str): Path to the PDF file
        max_pages (int): Maximum number of pages to process
        sample_rate (int): Process every nth page
        dpi (int): DPI for rendering
        
    Returns:
        list: List of PIL Image objects
    """
    images = []
    
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Get number of pages
        page_count = min(max_pages, pdf_document.page_count)
        logger.info(f"PDF has {pdf_document.page_count} pages, processing up to {page_count} pages with sample rate {sample_rate}")
        
        # Convert pages to images
        for page_num in range(0, page_count, sample_rate):
            try:
                page = pdf_document.load_page(page_num)
                
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                images.append(img)
                logger.debug(f"Converted page {page_num+1}")
            except Exception as e:
                logger.warning(f"Error converting page {page_num+1}: {str(e)}")
                continue
        
        # Close the PDF
        pdf_document.close()
        
        return images
    
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        return []
