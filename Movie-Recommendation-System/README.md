# Movie Recommendation System 🎬

## 📌 Overview
This project builds a movie recommendation system using content-based filtering. It suggests movies based on similarity in genres using machine learning techniques.

---

## 🎯 Objective
To develop a system that recommends movies to users based on their selected movie preferences.

---

## 🛠️ Tools & Technologies
- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## 📊 Dataset
Dataset used:
https://grouplens.org/datasets/movielens/

> Note: Dataset is not included due to size. Download `movies.csv` and place it in the project folder.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Loaded movie dataset  
- Cleaned and formatted genre data  

### 2. Feature Engineering
- Converted genres into text format  
- Applied CountVectorizer  

### 3. Similarity Calculation
- Used cosine similarity to measure movie similarity  

### 4. Recommendation System
- Built function to recommend top 5 similar movies  

### 5. Deployment
- Created Streamlit UI for user interaction  

---

## 📈 Results

The system successfully recommends similar movies based on genre similarity.

### 🔹 Output

![Output](ui_output.png)

---

## ⚠️ Note

The similarity matrix (`similarity.pkl`) is not included due to large file size.

To generate it, run the notebook: jupyter notebook movie_recommendation.ipynb


---

## 🧪 How to Run

- Clone Repository: git clone https://github.com/rohanronniie/Applied-AI-ML-Portfolio.git
- Navigate to Folder: cd Movie-Recommendation-System
- Install Requirements: pip install -r requirements.txt
- Run Notebook: jupyter notebook movie_recommendation.ipynb
- Run Application: python -m streamlit run app.py

---


---

## 📁 Project Structure

- movie_recommendation.ipynb  
- movies.pkl  
- app.py  
- ui_output.png  
- README.md  
- requirements.txt  

---

## 🚀 Conclusion
This project demonstrates how content-based filtering can be used to build an effective recommendation system. It highlights the use of similarity measures and machine learning techniques in real-world applications.

---

## 🔮 Future Improvements
- Include user-based collaborative filtering  
- Improve recommendation accuracy  
- Add movie posters and ratings  
- Deploy as a full web application  
