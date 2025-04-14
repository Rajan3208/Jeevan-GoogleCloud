import os
import pickle
import numpy as np
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

def train_spacy_model():
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    print("spaCy model loaded successfully")
    return nlp

def train_topic_model(sample_texts):
    print("Training topic model...")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(sample_texts)
    
    nmf_model = NMF(n_components=3, random_state=42)
    nmf_model.fit(dtm)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_filename = 'models/topic_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump((nmf_model, vectorizer), f)
    
    print(f"Topic model trained and saved to {model_filename}")
    return nmf_model, vectorizer

def train_sentiment_model():
    print("Creating sentiment analysis model...")
    # Create a simple keras model for sentiment classification
    model = Sequential([
        Dense(128, activation='relu', input_shape=(768,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    keras_model_path = 'models/sentiment_model.keras'
    model.save(keras_model_path)
    
    print(f"Sentiment model created and saved to {keras_model_path}")
    return model

def download_and_save_transformers_models():
    print("Downloading and caching transformer models...")
    # Download and cache the models
    sentiment_analyzer = pipeline('sentiment-analysis')
    summarizer = pipeline('summarization')
    
    # Create directory for transformer models if it doesn't exist
    os.makedirs('models/transformers', exist_ok=True)
    
    print("Transformer models downloaded and cached")

def main():
    print("Starting model training process...")
    
    # Train and save spaCy model
    nlp = train_spacy_model()
    
    # Sample texts for topic modeling training
    sample_texts = [
        "This document describes various legal provisions and regulations. It includes sections about compliance and legal requirements.",
        "The financial report outlines quarterly earnings, profit margins, and investment strategies. It discusses market trends and financial forecasts.",
        "This technical manual covers hardware specifications, software requirements, and troubleshooting procedures. It includes diagrams and technical details."
    ]
    
    # Train and save topic model
    topic_model, vectorizer = train_topic_model(sample_texts)
    
    # Train and save sentiment model
    sentiment_model = train_sentiment_model()
    
    # Download and cache transformer models
    download_and_save_transformers_models()
    
    print("All models trained and saved successfully!")

if __name__ == "__main__":
    main()