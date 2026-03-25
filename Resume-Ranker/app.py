import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Resume Ranker")

job_desc = st.text_area("Enter Job Description")

resume1 = st.text_area("Resume 1")
resume2 = st.text_area("Resume 2")
resume3 = st.text_area("Resume 3")

if st.button("Rank Resumes"):
    resumes = [resume1, resume2, resume3]
    
    documents = [job_desc] + resumes
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)
    
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    
    results = sorted(zip(resumes, similarity), key=lambda x: x[1], reverse=True)
    
    for res, score in results:
        st.write(f"{res} → Score: {score:.2f}")