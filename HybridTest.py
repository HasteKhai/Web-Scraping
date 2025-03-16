import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import distance as levenshtein_distance
from rapidfuzz import fuzz
from metaphone import doublemetaphone
import gensim.downloader as api

# Load Pretrained Word2Vec Model
w2v_model = api.load("word2vec-google-news-300")

# Load Data
df = pd.read_csv("MainDataset.csv")
reference_real = pd.read_csv("Balanced_Real_Reference_List.csv")['Name'].tolist()
reference_fictional = pd.read_csv("Balanced_Fictional_Reference_List.csv")['Name'].tolist()

# Load TF-IDF Vectorizer (Character N-Grams)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
tfidf_matrix = vectorizer.fit_transform(df['Name'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray())

# 🔹 **Compute Similarity-Based Features**
def levenshtein(name, reference):
    if not reference:
        return np.nan
    return min(levenshtein_distance(name, ref) for ref in reference)

def fuzzy_match(name, reference):
    if not reference:
        return np.nan
    return max(fuzz.ratio(name, ref) for ref in reference)

def double_metaphone_match(name, reference_set):
    encoded_name = doublemetaphone(name)[0]
    return 1 if encoded_name in reference_set else 0

# 🔹 **Compute NLP-Based Features (Word2Vec)**
def get_w2v_embedding(name):
    words = name.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.random.uniform(-0.01, 0.01, 300)  # Random small values if name not in vocab

# Apply Functions to Dataset
df['levenshtein_real'] = df['Name'].apply(lambda x: levenshtein(x, reference_real))
df['levenshtein_fictional'] = df['Name'].apply(lambda x: levenshtein(x, reference_fictional))
df['fuzzy_real'] = df['Name'].apply(lambda x: fuzzy_match(x, reference_real))
df['fuzzy_fictional'] = df['Name'].apply(lambda x: fuzzy_match(x, reference_fictional))
df['double_metaphone_real'] = df['Name'].apply(lambda x: double_metaphone_match(x, reference_real))
df['double_metaphone_fictional'] = df['Name'].apply(lambda x: double_metaphone_match(x, reference_fictional))

# Convert Word2Vec embeddings into DataFrame
w2v_features = np.vstack(df['Name'].apply(get_w2v_embedding).values)
w2v_df = pd.DataFrame(w2v_features)

# Combine All Features
df = pd.concat([df, tfidf_df, w2v_df], axis=1)

# Drop Unused Columns
df.drop(columns=['Name'], inplace=True)

# Define Features & Labels
X = df.drop(columns=['Label'])
y = df['Label']
X.columns = X.columns.astype(str)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Hybrid Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'fictional_name_hybrid_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Hybrid Model Trained & Saved!")

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import distance as levenshtein_distance
from rapidfuzz import fuzz
from metaphone import doublemetaphone
import gensim.downloader as api

# Load Models
model = joblib.load('fictional_name_hybrid_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
w2v_model = api.load("word2vec-google-news-300")

# Helper Functions
def levenshtein(name, reference):
    if not reference:
        return np.nan
    return min(levenshtein_distance(name, ref) for ref in reference)

def fuzzy_match(name, reference):
    if not reference:
        return np.nan
    return max(fuzz.ratio(name, ref) for ref in reference)

def double_metaphone_match(name, reference_set):
    encoded_name = doublemetaphone(name)[0]
    return 1 if encoded_name in reference_set else 0

def get_w2v_embedding(name):
    words = name.split()
    word_vectors = [w2v_model[word] for word in words if word in w2v_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.random.uniform(-0.01, 0.01, 300)  # Default small values

# Predict Function
def predict_fictionality(name):
    # Compute Features
    lev_real = levenshtein(name, reference_real)
    lev_fictional = levenshtein(name, reference_fictional)
    fuzzy_real = fuzzy_match(name, reference_real)
    fuzzy_fictional = fuzzy_match(name, reference_fictional)
    doublemetaphone_real = double_metaphone_match(name, reference_real)
    doublemetaphone_fict = double_metaphone_match(name, reference_fictional)
    w2v_vector = get_w2v_embedding(name).reshape(1, -1)
    tfidf_vector = vectorizer.transform([name]).toarray()

    # Merge Features
    features = np.hstack(([[lev_real, lev_fictional, fuzzy_real, fuzzy_fictional, doublemetaphone_real, doublemetaphone_fict]], w2v_vector, tfidf_vector))

    # Predict Probability
    prob_fictional = model.predict_proba(features)[0][1]

    print(f"\n📌 **Name Analysis:** {name}")
    print(f"🔹 Fictionality Probability: {prob_fictional:.2f}")

    return 'Fictional' if prob_fictional > 0.7 else 'Non-Fictional'

# Example Predictions
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
