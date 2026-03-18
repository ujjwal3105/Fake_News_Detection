import joblib
from preprocess import clean_text

model = joblib.load('../models/fake_news_model.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')

def predict_news(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    
    return "Real News" if prediction == 1 else "Fake News"
