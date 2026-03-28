# 📰 Fake News Detection

## 📌 Overview
This project detects whether a news article is **Real or Fake** using NLP and ML.

# Structure of the project
Fake_News_Detection
│
├── app
│   └── streamlit_app.py
│
├── data
│   └── fake_news_dataset.csv
│
├── models
│   └── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── notebooks
│   └── eda_analysis.ipynb
│
├── src
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

## 🚀 Features
- Text preprocessing
- TF-IDF vectorization
- Logistic Regression model
- Streamlit web app

## 🧠 Model
- Algorithm: Logistic Regression
- Accuracy: ~90% (depends on dataset)

## 🛠️ Installation
```bash
git clone <repo>
cd Fake_News_Detection
pip install -r requirements.txt
