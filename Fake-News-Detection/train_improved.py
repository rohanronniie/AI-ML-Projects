import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
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

# Paths
FAKE_CSV = 'data/Fake.csv'
TRUE_CSV = 'data/True.csv'
MODEL_PATH = 'models/model_improved.pkl'
VECTORIZER_PATH = 'models/vectorizer_improved.pkl'

print("Loading data...")
fake = pd.read_csv(FAKE_CSV)
real = pd.read_csv(TRUE_CSV)

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {df.shape}, Fake: {sum(df.label==0)}, Real: {sum(df.label==1)}")

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
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
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, ngram_range=(1,3), max_features=15000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("SMOTE balancing...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)

print(f"Balanced train: Fake {sum(y_train_bal==0)}, Real {sum(y_train_bal==1)}")

print("Training improved XGBoost...")
model = XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"\nImproved models saved to {MODEL_PATH} and {VECTORIZER_PATH}")

print("Test short Real headline:")
short_real = vectorizer.transform([clean_text("U.S., North Korea clash at U.N. arms forum on nuclear threat")])
print("Short Real pred:", model.predict(short_real)[0], model.predict_proba(short_real)[0])

print("Test full Real text:")
full_real = vectorizer.transform([clean_text("U.S., North Korea clash at U.N. arms forum on nuclear threat WASHINGTON (Reuters) - The United States and North Korea clashed...")])
print("Full Real pred:", model.predict(full_real)[0], model.predict_proba(full_real)[0])

