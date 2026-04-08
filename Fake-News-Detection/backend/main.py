
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from typing import Dict, Any

app = FastAPI(title="Fake News Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/model_improved.pkl"
VECTORIZER_PATH = "models/vectorizer_improved.pkl"

# Load once
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2])
    return text

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    real_probability: float
    fake_probability: float
    verdict: str
    confidence: float
    word_count: int
    threshold_used: float = 0.3

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    cleaned = clean_text(request.text)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    
    real_prob = float(prob[1])
    fake_prob = float(prob[0])
    pred = 1 if real_prob > 0.3 else 0
    verdict = "REAL" if pred == 1 else "FAKE"
    confidence = float(max(prob))
    
    return PredictResponse(
        real_probability=real_prob,
        fake_probability=fake_prob,
        verdict=verdict,
        confidence=confidence,
        word_count=len(cleaned.split())
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": "2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
</content>
</xai:function_call name="create_file">
<parameter name="absolute_path">d:/PROJECTS/Fake-News-Detection/backend/requirements.txt
