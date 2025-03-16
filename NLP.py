import pandas as pd
import numpy as np
import gensim.downloader as api
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("MainDataset.csv")

# Load Pre-trained Word2Vec Model (Google News 300D)
w2v_model = api.load("word2vec-google-news-300")  # 300D word vectors

# 🔥 **Feature Extraction with Word2Vec** 🔥
def get_w2v_embedding(name):
    words = name.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors for full name
    else:
        return np.zeros(300)  # Fallback if no words in vocabulary

df['w2v_embedding'] = df['Name'].apply(get_w2v_embedding)

# **TF-IDF Character N-Grams**
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
tfidf_matrix = vectorizer.fit_transform(df['Name'])

# Convert Word2Vec embeddings into DataFrame
w2v_features = np.vstack(df['w2v_embedding'].values)
w2v_df = pd.DataFrame(w2v_features, columns=[f'w2v_{i}' for i in range(w2v_features.shape[1])])

# Convert TF-IDF features into DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

# Merge All Features
df = pd.concat([df, w2v_df, tfidf_df], axis=1)
df.drop(columns=['w2v_embedding', 'Name'], inplace=True)

# Define Features & Labels
X = df.drop(columns=['Label'])
y = df['Label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model & TF-IDF Vectorizer
joblib.dump(model, 'fictional_name_w2v_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ NLP-based Word2Vec Model Trained & Saved!")
