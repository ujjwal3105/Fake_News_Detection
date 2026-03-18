import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import load_and_preprocess

def train():
    df = load_and_preprocess('../data/fake_news_dataset.csv')

    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, '../models/fake_news_model.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')

    print("Model trained and saved!")

if __name__ == "__main__":
    train()
