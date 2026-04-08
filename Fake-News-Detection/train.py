import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Paths (relative to project root)
FAKE_CSV = 'data/Fake.csv'
TRUE_CSV = 'data/True.csv'
MODEL_PATH = 'models/model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

print("Loading data...")
fake = pd.read_csv(FAKE_CSV)
real = pd.read_csv(TRUE_CSV)

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use title + text
df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

print(f"Dataset shape: {df.shape}, Fake: {sum(df.label==0)}, Real: {sum(df.label==1)}")

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special chars but keep punctuation for sentiment
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'<.*?>+', '', text)
    # Whitespace
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    # Tokenize, lemmatize, stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2])
    return text

print("Cleaning text...")
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Vectorizing...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Balancing classes with SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)

print(f"Balanced train: Fake {sum(y_train_bal==0)}, Real {sum(y_train_bal==1)}")

print("Training XGBoost...")
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"\nModels saved to {MODEL_PATH} and {VECTORIZER_PATH}")

