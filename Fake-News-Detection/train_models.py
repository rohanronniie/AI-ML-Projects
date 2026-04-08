"""
Fake News Detection Model Training Script
Trains Logistic Regression and Multinomial Naive Bayes.
Selects the best model and exports it under models/ along with the TF-IDF vectorizer.
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os

print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    Cleans text by removing HTML tags, URLs, special characters, and punctuation.
    Converts to lowercase and applies lemmatization.
    """
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    # Remove texts in brackets
    text = re.sub(r'\[.*?\]', '', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    # Remove newlines
    text = re.sub(r'\n', ' ', text)
    # Remove words with numbers
    text = re.sub(r'\w*\d\w*', ' ', text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize and remove stopwords + short words, then lemmatize
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(cleaned_tokens)

def main():
    print("Loading datasets...")
    # Load data
    try:
        fake_df = pd.read_csv('data/Fake.csv')
        true_df = pd.read_csv('data/True.csv')
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    print(f"Loaded {len(fake_df)} fake news and {len(true_df)} real news articles.")
    
    # Add labels
    fake_df['label'] = 0 # 0 for Fake
    true_df['label'] = 1 # 1 for Real
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    # Use ONLY the title for model training as per new logic requirement
    df['content'] = df['title']
    
    print("Cleaning text data... this might take a minute.")
    # Apply text cleaning
    df['clean_content'] = df['content'].apply(clean_text)
    
    print("Vectorizing data with TF-IDF...")
    # Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = tfidf_vectorizer.fit_transform(df['clean_content'])
    y = df['label']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model 1: Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    print(f"Logistic Regression -> Accuracy: {lr_acc:.4f}, F1-Score: {lr_f1:.4f}")
    
    # Model 2: Multinomial Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    
    nb_acc = accuracy_score(y_test, nb_preds)
    nb_f1 = f1_score(y_test, nb_preds)
    print(f"Multinomial NB -> Accuracy: {nb_acc:.4f}, F1-Score: {nb_f1:.4f}")
    
    # Select Best Model
    if lr_f1 >= nb_f1:
        print("\nSelecting Logistic Regression as the best model.")
        best_model = lr_model
        print("Confusion Matrix for Logistic Regression:")
        print(confusion_matrix(y_test, lr_preds))
    else:
        print("\nSelecting Multinomial Naive Bayes as the best model.")
        best_model = nb_model
        print("Confusion Matrix for Multinomial NB:")
        print(confusion_matrix(y_test, nb_preds))
    
    # Save the Models
    os.makedirs('models', exist_ok=True)
    print("\nExporting the model and vectorizer...")
    joblib.dump(best_model, 'models/model.pkl')
    joblib.dump(tfidf_vectorizer, 'models/vectorizer.pkl')
    print("Done! Models saved to models/model.pkl and models/vectorizer.pkl.")

if __name__ == "__main__":
    main()
