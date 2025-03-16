from FictionalClassification import levenshtein, fuzzy_match, soundex_match
import joblib
import pandas as pd

# Load model and references
model = joblib.load('fictional_name_classifier.pkl')
reference_real = joblib.load('reference_real.pkl')
reference_fictional = joblib.load('reference_fictional.pkl')


def predict_fictionality(name):
    lev_real = levenshtein(name, reference_real)
    lev_fictional = levenshtein(name, reference_fictional)
    fuzzy_real = fuzzy_match(name, reference_real)
    fuzzy_fictional = fuzzy_match(name, reference_fictional)
    soundex_real = soundex_match(name, reference_real)
    soundex_fictional = soundex_match(name, reference_fictional)

    # Ensure features are a DataFrame with the correct column names
    features = pd.DataFrame([[lev_real, lev_fictional, fuzzy_real, fuzzy_fictional, soundex_real, soundex_fictional]],
                            columns=['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
                                     'soundex_real', 'soundex_fictional'])

    # Predict with RandomForestClassifier
    prediction = model.predict(features)
    print("\n📌 **Name Analysis:**", name)
    print(f"🔹 Levenshtein Distance (Real):       {lev_real}")
    print(f"🔹 Levenshtein Distance (Fictional):  {lev_fictional}")
    print(f"🔹 Fuzzy Matching (Real):            {fuzzy_real}")
    print(f"🔹 Fuzzy Matching (Fictional):       {fuzzy_fictional}")
    print(f"🔹 Soundex Match (Real):             {'✅ Match' if soundex_real else '❌ No Match'}")
    print(f"🔹 Soundex Match (Fictional):        {'✅ Match' if soundex_fictional else '❌ No Match'}")
    return 'Fictional' if prediction[0] == 1 else 'Non-Fictional'


# Example Prediction
print(predict_fictionality("Froddo Baggins"))
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
