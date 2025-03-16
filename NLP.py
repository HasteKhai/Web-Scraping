import pandas as pd
import numpy as np
import fasttext
import fasttext.util
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv("MainDataset.csv")

# Download FastText model (300D embeddings)
fasttext.util.download_model('en', if_exists='ignore')  # English FastText
ft_model = fasttext.load_model('cc.en.300.bin')  # Load pre-trained model

# 🔥 Feature Extraction 🔥

# 1️⃣ **FastText Embeddings**
def get_fasttext_embedding(name):
    return np.mean([ft_model.get_word_vector(char) for char in name], axis=0)

df['fasttext_embedding'] = df['Name'].apply(get_fasttext_embedding)

# 2️⃣ **TF-IDF Character N-Grams**
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
tfidf_matrix = vectorizer.fit_transform(df['Name'])

# Convert FastText embeddings into a DataFrame
fasttext_features = np.vstack(df['fasttext_embedding'].values)
fasttext_df = pd.DataFrame(fasttext_features, columns=[f'fasttext_{i}' for i in range(fasttext_features.shape[1])])

# Convert TF-IDF features into a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

# Merge All Features
df = pd.concat([df, fasttext_df, tfidf_df], axis=1)
df.drop(columns=['fasttext_embedding', 'Name'], inplace=True)

# Define Features & Labels
X = df.drop(columns=['Label'])
y = df['Label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'fictional_name_nlp_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ NLP-based Model Trained & Saved!")
