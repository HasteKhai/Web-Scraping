import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from rapidfuzz import process, fuzz
import joblib
from metaphone import doublemetaphone
import nltk
from nltk.corpus import words
import swifter
from nltk.tokenize import word_tokenize
from rapidfuzz.distance import Levenshtein

df = pd.read_csv('MainDataset.csv')
balanced_Real_Reference = pd.read_csv('Balanced_Real_Reference_List.csv')
balanced_Fictional_Reference = pd.read_csv('Balanced_Fictional_Reference_List.csv')

reference_real = balanced_Real_Reference['Name'].tolist()
reference_fictional = balanced_Fictional_Reference['Name'].tolist()




def levenshtein(name, reference):
    # Find best match and its similarity score
    best_match, score, _ = process.extractOne(name, reference, scorer=fuzz.ratio)
    # Compute Levenshtein distance
    lev_distance = Levenshtein.distance(name, best_match)

    return lev_distance


df['levenshtein_real'] = df['Name'].swifter.apply(lambda x: levenshtein(x, reference_real))
df['levenshtein_fictional'] = df['Name'].swifter.apply(lambda x: levenshtein(x, reference_fictional))


def fuzzy_match(name, reference):
    # Find best match and its similarity score
    best_match, score, _ = process.extractOne(name, reference, scorer=fuzz.ratio)

    return score


df['fuzzy_real'] = df['Name'].swifter.progress_bar(True).apply(lambda x: fuzzy_match(x, reference_real))
df['fuzzy_fictional'] = df['Name'].swifter.progress_bar(True).apply(lambda x: fuzzy_match(x, reference_fictional))

# Precompute Metaphone for all reference names
reference_real_metaphone = {name: doublemetaphone(name)[0] for name in reference_real}
reference_fictional_metaphone = {name: doublemetaphone(name)[0] for name in reference_fictional}


def double_metaphone_match(name, reference_set):
    encoding = doublemetaphone(name)[0]  # Only use the primary encoding
    return 1 if encoding in reference_set.values() else 0


df['double_metaphone_real'] = (df['Name'].swifter.progress_bar(True).
                               apply(lambda x: double_metaphone_match(x, reference_real_metaphone)))
df['double_metaphone_fictional'] = (df['Name'].swifter.progress_bar(True).
                                    apply(lambda x: double_metaphone_match(x, reference_fictional_metaphone) * 5))

nltk.download('words')
english_words = set(words.words())

nltk.download('punkt_tab')


# Tokenization better than .split(), avoiding issues with splitting punctuations and special characters
def is_dictionary_word(name):
    tokens = word_tokenize(name.lower())
    return int(any(word in english_words for word in tokens))


df['is_dictionary_word'] = df['Name'].apply(is_dictionary_word)

df['double_metaphone_real'] *= 5
df['double_metaphone_fictional'] *= 3
df['fuzzy_real'] *= 0.85
df['fuzzy_fictional'] *= 0.75
df['is_dictionary_word'] = df['is_dictionary_word'] * df['fuzzy_fictional']

# GirdSearch on RF
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [10, 20, 30, None],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples at leaf node
}

X = df[['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
        'double_metaphone_real', 'double_metaphone_fictional', 'is_dictionary_word']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize GridSearchCV
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
#                            param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearch to find the best model
# grid_search.fit(X_train, y_train)

# Print best parameters
# print("🔹 Best Hyperparameters:", grid_search.best_params_)

# Get the best model
# model = grid_search.best_estimator_
model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=4, min_samples_split=2, random_state=42,
                               n_jobs=-1)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
from sklearn.metrics import classification_report, confusion_matrix

print("📌 Classification Report:\n", classification_report(y_test, y_pred))
print("📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'fictional_name_classifier.pkl')
joblib.dump(reference_real, 'reference_real.pkl')
joblib.dump(reference_fictional, 'reference_fictional.pkl')
joblib.dump(reference_real_metaphone, 'reference_real_metaphone.pkl')
joblib.dump(reference_fictional_metaphone, 'reference_fictional_metaphone.pkl')

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=10)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
