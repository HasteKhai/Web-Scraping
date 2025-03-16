from FictionalClassification import levenshtein, fuzzy_match, soundex_match
import joblib
import pandas as pd

# Load model and references
model = joblib.load('fictional_name_classifier.pkl')
reference_real = joblib.load('reference_real.pkl')
print(len(reference_real))
reference_fictional = joblib.load('reference_fictional.pkl')
print(len(reference_fictional))
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
    print(lev_real, lev_fictional, fuzzy_real, fuzzy_fictional)
    return 'Fictional' if prediction[0] == 1 else 'Non-Fictional'

# Example Prediction
print(predict_fictionality("Froddo Baggins"))
