from FictionalClassification import levenshtein, fuzzy_match, double_metaphone_match, is_dictionary_word
import joblib
import pandas as pd

# Load model and references
model = joblib.load('fictional_name_classifier.pkl')
reference_real = joblib.load('reference_real.pkl')
reference_fictional = joblib.load('reference_fictional.pkl')
reference_real_metaphone = joblib.load('reference_real_metaphone.pkl')
reference_fictional_metaphone = joblib.load('reference_fictional_metaphone.pkl')

precomputed_features = pd.read_csv('precomputed_features.csv')
def predict_fictionality(name):
    row = precomputed_features[precomputed_features['Name'] == name]
    if not row.empty:
        features = row[['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real',
                        'fuzzy_fictional', 'double_metaphone_real', 'double_metaphone_fictional',
                        'is_dictionary_word']]
        prediction = model.predict_proba(features)[0][1]
    else:
        lev_real = levenshtein(name, reference_real)
        lev_fictional = levenshtein(name, reference_fictional)
        fuzzy_real = fuzzy_match(name, reference_real)
        fuzzy_fictional = fuzzy_match(name, reference_fictional)
        doublemetaphone_real = double_metaphone_match(name, reference_real_metaphone)
        doublemetaphone_fict = double_metaphone_match(name, reference_fictional_metaphone)
        dict_word = is_dictionary_word(name)

    # Ensure features are a DataFrame with the correct column names
    features = pd.DataFrame([[lev_real, lev_fictional, fuzzy_real, fuzzy_fictional, doublemetaphone_real,
                              doublemetaphone_fict, dict_word]],
                            columns=['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
                                     'double_metaphone_real', 'double_metaphone_fictional', 'is_dictionary_word'])

    # Get Probabilities
    prob_fictional = model.predict_proba(features)[0][1]

    # Adjusted Threshold (was 0.5)
    threshold = 0.8  # Increase threshold to reduce false positives
    prediction = 1 if prob_fictional > threshold else 0

    # Predict with RandomForestClassifier
    print("\n📌 **Name Analysis:**", name)
    print(f"🔹 Fict Prob:       {prob_fictional}")
    print(f"🔹 Levenshtein Distance (Real):       {lev_real}")
    print(f"🔹 Levenshtein Distance (Fictional):  {lev_fictional}")
    print(f"🔹 Fuzzy Matching (Real):            {fuzzy_real}")
    print(f"🔹 Fuzzy Matching (Fictional):       {fuzzy_fictional}")
    print(f"🔹 DMetaphone Match (Real):             {'✅ Match' if doublemetaphone_real else '❌ No Match'}")
    print(f"🔹 DMetaphone Match (Fictional):        {'✅ Match' if doublemetaphone_fict else '❌ No Match'}")
    print(f"🔹 Contains a Dictionnary Word:        {dict_word}")
    return 'Fictional' if prediction == 1 else 'Non-Fictional'


# Example Prediction
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

# Get feature importance scores
feature_importance = model.feature_importances_
feature_names = ['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
                 'double_metaphone_real', 'double_metaphone_fictional', 'is_dictionary_word']

# Sort feature importances in descending order
sorted_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

# Print feature importance scores
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")
