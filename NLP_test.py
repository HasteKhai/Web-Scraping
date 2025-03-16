import joblib
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Model & Data
model = joblib.load('fictional_name_w2v_model.pkl')  # Load trained model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load TF-IDF vectorizer

# Load Pre-trained Word2Vec Model
w2v_model = api.load("word2vec-google-news-300")  # 300D embeddings

# 🔥 **Extract Word2Vec Embeddings**
def get_w2v_embedding(name):
    words = name.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors for full name
    else:
        return np.zeros(300)  # Fallback if no words are in vocabulary

# 🔍 **Prediction Function**
def predict_fictionality(name):
    # Extract Features
    w2v_vector = get_w2v_embedding(name).reshape(1, -1)
    tfidf_vector = vectorizer.transform([name]).toarray()

    # Merge Features
    features = np.hstack((w2v_vector, tfidf_vector))

    # Predict Probability
    prob_fictional = model.predict_proba(features)[0][1]

    print(f"\n📌 **Name Analysis:** {name}")
    print(f"🔹 Fictionality Probability: {prob_fictional:.2f}")

    return 'Fictional' if prob_fictional > 0.7 else 'Non-Fictional'

# **Example Predictions**
print(predict_fictionality("Knee Ellen"))
print(predict_fictionality("Mickey Mouse"))
print(predict_fictionality("Mickey Mousse"))
print(predict_fictionality("Alexandre"))
print(predict_fictionality("Orange"))
print(predict_fictionality("Oranga"))
print(predict_fictionality("Dryad"))
print(predict_fictionality("Batman"))
print(predict_fictionality("Apple"))
print(predict_fictionality("Timothy Smay"))
print(predict_fictionality("Christina Perez"))
print(predict_fictionality("Alexandre Gagnon"))
print(predict_fictionality("Jack Sparrow"))
print(predict_fictionality("Eric Brault"))
print(predict_fictionality("Kane Yu-Kis Mi"))
print(predict_fictionality("Ai Wan Tyu"))
print(predict_fictionality("Youssef Hamza"))
print(predict_fictionality("Sonia Creo"))
print(predict_fictionality("khai Trinh"))
