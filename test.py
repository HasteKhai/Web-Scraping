from FictionalClassification import levenshtein, fuzzy_match, double_metaphone_match
import joblib
import pandas as pd

# Load model and references
model = joblib.load('fictional_name_classifier.pkl')
reference_real = joblib.load('reference_real.pkl')
reference_fictional = joblib.load('reference_fictional.pkl')
reference_real_metaphone = joblib.load('reference_real_metaphone.pkl')
reference_fictional_metaphone = joblib.load('reference_fictional_metaphone.pkl')
def predict_fictionality(name):
    lev_real = levenshtein(name, reference_real)
    lev_fictional = levenshtein(name, reference_fictional)
    fuzzy_real = fuzzy_match(name, reference_real)
    fuzzy_fictional = fuzzy_match(name, reference_fictional)
    doublemetaphone_real = double_metaphone_match(name, reference_real_metaphone)
    doublemetaphone_fict = double_metaphone_match(name, reference_fictional_metaphone)

    # Ensure features are a DataFrame with the correct column names
    features = pd.DataFrame([[lev_real, lev_fictional, fuzzy_real, fuzzy_fictional, doublemetaphone_real,
                              doublemetaphone_fict]],
                            columns=['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
                                     'double_metaphone_real', 'double_metaphone_fictional'])

    # Predict with RandomForestClassifier
    prediction = model.predict(features)
    print("\n📌 **Name Analysis:**", name)
    print(f"🔹 Levenshtein Distance (Real):       {lev_real}")
    print(f"🔹 Levenshtein Distance (Fictional):  {lev_fictional}")
    print(f"🔹 Fuzzy Matching (Real):            {fuzzy_real}")
    print(f"🔹 Fuzzy Matching (Fictional):       {fuzzy_fictional}")
    print(f"🔹 DMetaphone Match (Real):             {'✅ Match' if doublemetaphone_real else '❌ No Match'}")
    print(f"🔹 DMetaphone Match (Fictional):        {'✅ Match' if doublemetaphone_fict else '❌ No Match'}")
    return 'Fictional' if prediction[0] == 1 else 'Non-Fictional'


# Example Prediction
print(predict_fictionality("Knee"))
print(predict_fictionality("Mickey Mouse"))
print(predict_fictionality("Alexandre"))
print(predict_fictionality("Orange"))
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