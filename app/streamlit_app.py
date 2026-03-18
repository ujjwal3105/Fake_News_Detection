import streamlit as st
import joblib
import sys
import os

sys.path.append(os.path.abspath("../src"))

from preprocess import clean_text

# Load model
model = joblib.load('../models/fake_news_model.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')

st.title("📰 Fake News Detection App")

st.write("Enter a news article to check if it's real or fake.")

user_input = st.text_area("News Content")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success("✅ This is Real News")
        else:
            st.error("❌ This is Fake News")
