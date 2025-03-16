import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from Levenshtein import distance as levenshtein_distance
from rapidfuzz import fuzz
from jellyfish import soundex
import joblib
from metaphone import doublemetaphone

fictional_names = pd.read_csv("Fictional_Names.csv")
reference_fictional = fictional_names['Name'].tolist()
fictional_names['Label'] = 1  # Ref list for fictional names
df1 = pd.DataFrame(fictional_names)

real_names2 = pd.read_csv

real_names = pd.read_csv("Customer_Names.csv")
real_names['Name'] = real_names['First Name'] + " " + real_names['Last Name']
real_names = real_names[['Name']]
reference_real = real_names['Name'].tolist()  # Ref list for real names
real_names['Label'] = 0
df2 = pd.DataFrame(real_names)

df = pd.concat([fictional_names, real_names])


def levenshtein(name, reference):
    if not reference:
        return np.nan
    return min(levenshtein_distance(name, ref) for ref in reference)


df['levenshtein_real'] = df['Name'].apply(lambda x: levenshtein(x, reference_real))
df['levenshtein_fictional'] = df['Name'].apply(lambda x: levenshtein(x, reference_fictional))


def fuzzy_match(name1, reference):
    if not reference:
        return np.nan
    return max(fuzz.ratio(name1, ref) for ref in reference)


df['fuzzy_real'] = df['Name'].apply(lambda x: fuzzy_match(x, reference_real))
df['fuzzy_fictional'] = df['Name'].apply(lambda x: fuzzy_match(x, reference_fictional))


# Precompute Metaphone for all reference names
reference_real_metaphone = {name: doublemetaphone(name)[0] for name in reference_real}
reference_fictional_metaphone = {name: doublemetaphone(name)[0] for name in reference_fictional}

def double_metaphone_match(name, reference_set):
    encoded_name = doublemetaphone(name)[0]  # Compute once
    return 1 if encoded_name in reference_set.values() else 0  # Fast lookup

df['double_metaphone_real'] = df['Name'].apply(lambda x: double_metaphone_match(x, reference_real_metaphone))
df['double_metaphone_fictional'] = df['Name'].apply(lambda x: double_metaphone_match(x, reference_fictional_metaphone))


X = df[['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
        'double_metaphone_real', 'double_metaphone_fictional']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

joblib.dump(model, 'fictional_name_classifier.pkl')
joblib.dump(reference_real, 'reference_real.pkl')
joblib.dump(reference_fictional, 'reference_fictional.pkl')
joblib.dump(reference_real_metaphone, 'reference_real_metaphone.pkl')
joblib.dump(reference_fictional_metaphone, 'reference_fictional_metaphone.pkl')
