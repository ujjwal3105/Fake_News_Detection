import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    # assuming columns: text, label
    df = df[['text', 'label']]
    df['text'] = df['text'].apply(clean_text)
    
    return df
