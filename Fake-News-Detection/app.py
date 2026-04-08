"""
Fake News Detection Web Application
Developed using Streamlit, providing real-time text classification from News Titles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os
import time
import altair as alt

# Options
MODEL_PATH = 'models/model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

@st.cache_resource(show_spinner=False)
def load_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def clean_text(text: str) -> str:
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

def get_top_keywords(vec, vectorizer, top_n=5) -> list:
    feature_names = vectorizer.get_feature_names_out()
    coo_matrix = vec.tocoo()
    tuples = zip(coo_matrix.col, coo_matrix.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    top_keywords = []
    for idx, float_val in sorted_items[:top_n]:
        top_keywords.append({"Token": feature_names[idx], "Weight": float_val})
    return top_keywords

def main():
    st.set_page_config(page_title="TruthLens AI", page_icon="🕵️", layout="wide")
    
    # Safe and extremely modern CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;700;800&display=swap');

        /* Explicitly targeting our custom elements to safeguard Streamlit's internal icons (like expander arrows) */
        .title-hero {
            font-family: 'Manrope', sans-serif;
            text-align: center;
            padding-bottom: 2rem;
            animation: fadeIn 1s;
        }
        
        .main-gradient-text {
            font-size: 3.5rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        
        .subtitle-text {
            color: #9CA3AF;
            font-size: 1.2rem;
            font-weight: 400;
            margin-top: 5px;
        }

        .auth-box-real {
            background-color: rgba(16, 185, 129, 0.1);
            border-left: 5px solid #10B981;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Manrope', sans-serif;
        }

        .auth-box-fake {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 5px solid #EF4444;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Manrope', sans-serif;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Hero Title
    st.markdown("""
        <div class="title-hero">
            <div class="main-gradient-text">TruthLens AI</div>
            <div class="subtitle-text">Syntactic Pattern Intelligence for News Headlines</div>
        </div>
    """, unsafe_allow_html=True)
    
    model, vectorizer = load_resources()
    if model is None or vectorizer is None:
        st.error("⚠️ Model Artifacts not found. Run Python training script backend first.")
        st.stop()
    
    # Input Area - Centered and Sleek
    _, center_col, _ = st.columns([1, 6, 1])
    with center_col:
        headline = st.text_input(
            label="Input target headline",
            label_visibility="collapsed",
            placeholder="Paste headline here (e.g., 'BREAKING: New technology changes everything')..."
        )
        
        btn = st.button("EXECUTE NEURAL SCAN", use_container_width=True)

    if btn:
        if not headline or len(headline.strip()) < 5:
            st.toast("Headline too short for reliable analysis.", icon="⚠️")
        else:
            with center_col:
                with st.spinner("Processing semantics..."):
                    time.sleep(0.4) 
                    cleaned = clean_text(headline)
                    if len(cleaned.split()) == 0:
                        st.error("⚠️ Filtered to empty words. Add valid english words.")
                        st.stop()
                         
                    vec = vectorizer.transform([cleaned])
                    pred = model.predict(vec)[0]
                    prob = model.predict_proba(vec)[0]
                    
                    is_real = True if pred == 1 else False
                    conf = max(prob)
                    
                    st.write("---")
                    
                    # Dashboard Layout
                    out_col1, out_col2 = st.columns([1, 1.2])
                    
                    with out_col1:
                        st.write("### Diagnostics")
                        if is_real:
                            st.markdown(f"""
                            <div class="auth-box-real">
                                <h2 style='color:#34D399; margin:0;'>AUTHENTIC</h2>
                                <p style='color:#E5E7EB; margin-top:5px; margin-bottom:0;'>Organic linguistic structure detected.</p>
                                <h1 style='color:#white; margin-top:10px;'>{conf*100:.1f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="auth-box-fake">
                                <h2 style='color:#F87171; margin:0;'>FABRICATED</h2>
                                <p style='color:#E5E7EB; margin-top:5px; margin-bottom:0;'>Sensational / Clickbait phrasing detected.</p>
                                <h1 style='color:#white; margin-top:10px;'>{conf*100:.1f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)

                    with out_col2:
                        st.write("### Probability Distribution")
                        # Create a professional chart using Altair instead of raw numbers
                        chart_data = pd.DataFrame({
                            "Status": ["Fake Probability", "Real Probability"],
                            "Percentage": [prob[0] * 100, prob[1] * 100]
                        })
                        
                        donut = alt.Chart(chart_data).mark_arc(innerRadius=40).encode(
                            theta="Percentage",
                            color=alt.Color("Status", scale=alt.Scale(
                                domain=['Fake Probability', 'Real Probability'],
                                range=['#EF4444', '#10B981']
                            )),
                            tooltip=["Status", alt.Tooltip("Percentage", format=".1f")]
                        ).properties(height=200).interactive()
                        
                        st.altair_chart(donut, use_container_width=True)

                    # Tokens Graph
                    top_words_data = get_top_keywords(vec, vectorizer, top_n=6)
                    
                    with st.expander("🔍 Neural Decision Insight (TF-IDF Trace)", expanded=True):
                        st.markdown("Top semantic indicators determining the final prediction:")
                        if top_words_data:
                            # Plot Horizontal Bar chart
                            df_kw = pd.DataFrame(top_words_data)
                            bar_chart = alt.Chart(df_kw).mark_bar(cornerRadiusEnd=4, height=15).encode(
                                x=alt.X('Weight:Q', title="Aesthetic Gravity (TF-IDF)",
                                        scale=alt.Scale(domain=[0, df_kw['Weight'].max() * 1.2]) # Add padding
                                       ),
                                y=alt.Y('Token:N', sort='-x', title=""),
                                color=alt.Color('Weight:Q', scale=alt.Scale(scheme='teals'), legend=None),
                                tooltip=["Token", "Weight"]
                            ).properties(height=200 + (len(df_kw) * 20))
                            
                            st.altair_chart(bar_chart, use_container_width=True)
                        else:
                            st.info("No standout tokens resolved.")

    st.markdown("<br><hr style='opacity: 0.2;'><div style='text-align: center; color: #6B7280; font-size: 0.8rem;'>Powered by Advanced Scikit-Learn Ecosystem</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
