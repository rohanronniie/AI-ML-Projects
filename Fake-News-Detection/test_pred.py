import sys
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

MODEL_PATH = 'models/model_improved.pkl'
VECTORIZER_PATH = 'models/vectorizer_improved.pkl'

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\[. *?\\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\\w*\\d\\w*', ' ', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2])
    return text

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_text = ' '.join(sys.argv[1:])
        cleaned = clean_text(input_text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        print(f"Prediction: {'Real' if pred == 1 else 'Fake'}")
        print(f"Prob Real: {prob[1]:.2%}, Prob Fake: {prob[0]:.2%}")
    else:
        print("Usage: python test_pred.py \"your news title and text here\"")

