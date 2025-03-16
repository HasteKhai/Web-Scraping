import joblib
import pandas as pd
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Model & Data
model = joblib.load('fictional_name_nlp_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
ft_model = fasttext.load_model('cc.en.300.bin')

# Extract Features
def get_fasttext_embedding(name):
    return np.mean([ft_model.get_word_vector(char) for char in name], axis=0)

# Predict Function
def predict_fictionality(name):
    # Extract Features
    fasttext_vector = get_fasttext_embedding(name).reshape(1, -1)
    tfidf_vector = vectorizer.transform([name]).toarray()

    # Merge Features
    features = np.hstack((fasttext_vector, tfidf_vector))

    # Predict Probability
    prob_fictional = model.predict_proba(features)[0][1]

    print(f"\n📌 **Name Analysis:** {name}")
    print(f"🔹 Fictionality Probability: {prob_fictional:.2f}")

    return 'Fictional' if prob_fictional > 0.7 else 'Non-Fictional'

# Example Predictions
print(predict_fictionality("Knee Ellen"))
print(predict_fictionality("Mickey Mouse"))
print(predict_fictionality("Alexandre"))
print(predict_fictionality("Batman"))
