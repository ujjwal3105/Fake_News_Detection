import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords

# ✅ Download stopwords (safe for cloud)
nltk.download('stopwords')

# ================= PREPROCESS FUNCTION =================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ================= LOAD MODEL =================
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "..", "models", "fake_news_model.pkl")
vectorizer_path = os.path.join(current_dir, "..", "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ================= UI =================
st.title("📰 Fake News Detection App")

st.write("Enter a news article to check if it is Real or Fake")

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
