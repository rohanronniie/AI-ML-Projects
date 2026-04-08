import pandas as pd
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Exact functions from train.py
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

MODEL_PATH = 'models/model_improved.pkl'
VECTORIZER_PATH = 'models/vectorizer_improved.pkl'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def clean_text(text):
    # Exact from train.py
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    # Remove special chars but keep punctuation for sentiment
    text = re.sub(r'\\[. *?\\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'<.*?>+', '', text)
    # Whitespace
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\w*\\d\\w*', ' ', text)
    # Tokenize, lemmatize, stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2])
    return text

# Test samples
fake_sample = "Donald Trump Sends Out Embarrassing New Year's Eve Message"
real_sample = "Local store owner celebrates 50 years in business"

print("Fake sample clean:", clean_text(fake_sample))
print("Real sample clean:", clean_text(real_sample))

fake_vec = vectorizer.transform([clean_text(fake_sample)])
real_vec = vectorizer.transform([clean_text(real_sample)])

print("Fake pred:", model.predict(fake_vec)[0], model.predict_proba(fake_vec)[0])
print("Real pred:", model.predict(real_vec)[0], model.predict_proba(real_vec)[0])

print("Pipeline test complete. Matches train.py preprocess.")

