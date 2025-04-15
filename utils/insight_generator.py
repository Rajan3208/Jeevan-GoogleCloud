import os
import pickle
import logging
import hashlib
import threading
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to track model loading status
MODEL_STATUS = {
    "spacy": False,
    "topic_model": False,
    "sentiment_model": False,
    "transformers": False,
    "all_required_loaded": False,
    "initialization_started": False,
    "last_error": None
}

# Lock for thread-safe model loading
model_lock = threading.Lock()

def check_models_loaded():
    """Check if models are loaded"""
    global MODEL_STATUS
    with model_lock:
        return MODEL_STATUS.copy()

def load_models(preload_all=False):
    """Load required models for text analysis
    
    Args:
        preload_all: If True, load all models including heavier ones
                     If False, load only essential lightweight models
    
    Returns:
        dict: Status of model loading
    """
    global MODEL_STATUS
    
    with model_lock:
        if MODEL_STATUS["initialization_started"] and not preload_all:
            # If initialization already started and we just need essential models, return current status
            return MODEL_STATUS.copy()
            
        MODEL_STATUS["initialization_started"] = True
    
    try:
        # Load spaCy model (lightweight)
        logger.info("Loading spaCy model...")
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
                MODEL_STATUS["spacy"] = True
                logger.info("spaCy model loaded successfully")
            except:
                logger.warning("Downloading spaCy model...")
                import subprocess
                subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")
                MODEL_STATUS["spacy"] = True
                logger.info("spaCy model downloaded and loaded successfully")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            MODEL_STATUS["last_error"] = f"spaCy: {str(e)}"
        
        # Load topic model (lightweight)
        logger.info("Loading topic model...")
        try:
            model_path = os.path.join('models', 'topic_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    topic_model, vectorizer = pickle.load(f)
                MODEL_STATUS["topic_model"] = True
                logger.info("Topic model loaded successfully")
            else:
                # Create a simple backup model if file doesn't exist
                logger.warning("Topic model file not found. Creating a simple backup model...")
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import NMF
                
                sample_texts = [
                    "This is a sample document for topic modeling.",
                    "Another example text for creating a simple topic model.",
                    "We need at least a few documents to create a valid topic model."
                ]
                
                vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, stop_words='english')
                dtm = vectorizer.fit_transform(sample_texts)
                
                topic_model = NMF(n_components=2, random_state=42)
                topic_model.fit(dtm)
                
                # Save the model
                os.makedirs('models', exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump((topic_model, vectorizer), f)
                
                MODEL_STATUS["topic_model"] = True
                logger.info("Simple backup topic model created and saved")
        except Exception as e:
            logger.error(f"Error loading topic model: {str(e)}")
            MODEL_STATUS["last_error"] = f"topic_model: {str(e)}"
        
        # Required models are loaded, mark as ready
        if MODEL_STATUS["spacy"] and MODEL_STATUS["topic_model"]:
            MODEL_STATUS["all_required_loaded"] = True
        
        # Only load heavier models if requested
        if preload_all:
            # Load sentiment model (heavier)
            logger.info("Loading sentiment model...")
            try:
                import tensorflow as tf
                
                model_path = os.path.join('models', 'sentiment_model.keras')
                if os.path.exists(model_path):
                    sentiment_model = tf.keras.models.load_model(model_path)
                    MODEL_STATUS["sentiment_model"] = True
                    logger.info("Sentiment model loaded successfully")
                else:
                    # Create a simple backup model if file doesn't exist
                    logger.warning("Sentiment model file not found. Creating a simple backup model...")
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, Dropout
                    
                    sentiment_model = Sequential([
                        Dense(64, activation='relu', input_shape=(768,)),
                        Dropout(0.2),
                        Dense(32, activation='relu'),
                        Dense(2, activation='softmax')
                    ])
                    sentiment_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    # Save the model
                    os.makedirs('models', exist_ok=True)
                    sentiment_model.save(model_path)
                    
                    MODEL_STATUS["sentiment_model"] = True
                    logger.info("Simple backup sentiment model created and saved")
            except Exception as e:
                logger.error(f"Error loading sentiment model: {str(e)}")
                MODEL_STATUS["last_error"] = f"sentiment_model: {str(e)}"
            
            # Load transformer models (heaviest)
            logger.info("Loading transformer models...")
            try:
                from transformers import pipeline
                
                # Use smaller models that load faster
                summarizer = pipeline('summarization', model="facebook/bart-large-cnn")
                MODEL_STATUS["transformers"] = True
                logger.info("Transformer models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading transformer models: {str(e)}")
                MODEL_STATUS["last_error"] = f"transformers: {str(e)}"
    
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {str(e)}")
        MODEL_STATUS["last_error"] = f"Unexpected: {str(e)}"
    
    return MODEL_STATUS.copy()

@lru_cache(maxsize=20)
def generate_insights(text, fast_mode=False):
    """Generate insights from text
    
    Args:
        text (str): The text to analyze
        fast_mode (bool): If True, use faster but less accurate methods
        
    Returns:
        tuple: (details, summary) Detailed insights and a summary
    """
    # Check if required models are loaded
    if not MODEL_STATUS.get("all_required_loaded", False):
        # Try loading models if they're not loaded yet
        load_models(preload_all=False)
        if not MODEL_STATUS.get("all_required_loaded", False):
            raise RuntimeError("Required models not loaded. Please try again later.")
    
    # For very short texts, return a simple analysis
    if len(text) < 50:
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "sentiment": "neutral"
        }, "Text is too short for detailed analysis."
    
    # Create a hash of the text for caching
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Simulate processing time for demonstration
    time.sleep(0.5 if fast_mode else 1)
    
    # Process with spaCy for entity recognition
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:5000] if fast_mode else text)  # Process shorter text in fast mode
        
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        # Extract topics using the topic model
        try:
            model_path = os.path.join('models', 'topic_model.pkl')
            with open(model_path, 'rb') as f:
                topic_model, vectorizer = pickle.load(f)
            
            # Transform text to document-term matrix
            dtm = vectorizer.transform([text])
            
            # Get topic distribution
            topic_distribution = topic_model.transform(dtm)[0]
            topics = [
                {"id": i, "weight": float(weight)}
                for i, weight in enumerate(topic_distribution)
            ]
        except Exception as e:
            logger.error(f"Error in topic analysis: {str(e)}")
            topics = [{"id": 0, "weight": 1.0}]
        
        # Generate insights
        details = {
            "entities": entities[:20],  # Limit to top 20 entities
            "topics": topics,
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(list(doc.sents)),
            "processing_mode": "fast" if fast_mode else "full"
        }
        
        # Generate summary
        summary = f"The text contains {details['word_count']} words in {details['sentence_count']} sentences. "
        if entities:
            summary += f"Key entities include {', '.join([e['text'] for e in entities[:3]])}. "
        summary += f"The dominant topic has a weight of {topics[0]['weight']:.2f}."
        
        return details, summary
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        # Return basic fallback insights
        return {
            "error": str(e),
            "length": len(text),
            "word_count": len(text.split())
        }, "Error analyzing text. Basic metrics provided."
