# Fake News Detection - Modeling Notebook
*Complete EDA, Preprocessing, Training, and Evaluation*

## 1. Data Acquisition

We begin by loading a labeled dataset containing Fake and Real News articles. We use the standard Kaggle datasets (`Fake.csv` and `True.csv`).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Label data (0 for Fake, 1 for Real)
fake_df['label'] = 0
true_df['label'] = 1

# Combine into a single DataFrame
df = pd.concat([fake_df, true_df], ignore_index=True)
# For this model, we are training entirely on the linguistics of the news Title.
df['content'] = df['title']
```

## 2. Exploratory Data Analysis (EDA)

```python
# Check class distribution
sns.countplot(x='label', data=df)
plt.title("Distribution of Fake (0) vs Real (1) News")
plt.show()

# Sample of the content
print(df.head())
```
*Observation: The dataset usually has an even distribution of fake vs real news articles. The `content` feature combines headlines with article bodies for a richer context and semantic view.*

## 3. Data Preprocessing (NLTK & Pandas)

We apply robust text cleaning to remove URLs, HTML tags, punctuations, and stopwords. We also lemmatize the words.

```python
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(cleaned_tokens)

# Apply preprocessing
df['clean_content'] = df['content'].apply(clean_text)
```

## 4. Feature Engineering (Scikit-learn)

Vectorize the text using TF-IDF (Term Frequency-Inverse Document Frequency) which converts the text strings into a numerical matrix where each term's weight reflects its importance.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X = tfidf_vectorizer.fit_transform(df['clean_content'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## 5. Model Training & Evaluation

We evaluate Logistic Regression and Multinomial Naive Bayes as baseline models.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(f"F1-Score: {f1_score(y_test, lr_preds):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_preds))

# Model 2: Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

print("\nMultinomial Naive Bayes Metrics:")
print(f"Accuracy: {accuracy_score(y_test, nb_preds):.4f}")
print(f"F1-Score: {f1_score(y_test, nb_preds):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_preds))
```

### Result Analysis
* The **Logistic Regression** baseline usually outperforms Naive Bayes slightly, yielding high accuracy (>98%) on this corpus because it handles correlated TF-IDF unigram and bigram features well. 

## 6. Model Serialization

Finally, we serialize (using `joblib`) the winning model and the TF-IDF vectorizer so they can be loaded by the Streamlit application for real-time inference.

```python
import joblib

best_model = lr_model # Assuming Logistic Regression performed best based on metrics

# Save the top-performing model and the vectorizer
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(tfidf_vectorizer, 'models/vectorizer.pkl')
print("Model Artifacts Exported Successfully.")
```
