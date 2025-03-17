import pandas as pd
from itertools import product
import random

# Load fictional names
fictional_names = pd.read_csv("Fictional_Names.csv")
fictional_names['Label'] = 1

# Balance the fictional reference list
reference_fictional = fictional_names['Name'].sample(n=5000, random_state=42).tolist()
reference_fictional.extend(fictional_names['Name'].head(5000).tolist())

# Load real names
first_names = pd.read_csv("common-forenames.csv")
surnames = pd.read_csv("common-surnames-by-country.csv").dropna(subset=['Romanized Name'])
external = pd.read_csv('Customer_Names.csv')
first_names['Romanized Name'] = first_names['Romanized Name'].str.lower()
surnames['Romanized Name'] = surnames['Romanized Name']

merged_df = first_names.merge(surnames, on='Country', suffixes=('_first', '_last'))
# Generate combinations of matching names
all_combinations = list(zip(merged_df['Romanized Name_first'], merged_df['Romanized Name_last']))
all_combinations2 = list(product(external['First Name'].str.lower(), external['Last Name'].str.lower()))
# Shuffle to introduce randomness
random.shuffle(all_combinations)
random.shuffle(all_combinations2)

# Convert back to DataFrame if needed
combinations_df = pd.DataFrame(all_combinations, columns=['First Name', 'Surname'])
combinations_df['Name'] = (combinations_df['First Name'] + " " + combinations_df['Surname'])


# Convert combinations to DataFrame
external_names = pd.DataFrame(all_combinations2, columns=['First Name', 'Last Name'])
external_names['Name'] = external_names['First Name'] + " " + external_names['Last Name']

# Sample equal numbers for real reference list
reference_real = combinations_df['Name'].sample(n=5000, random_state=42).tolist()
reference_real.extend(external_names['Name'].sample(n=5000, random_state=42).tolist())  # Use .extend()

# Real names dataset
real_names = combinations_df[['Name']].sample(n=10000, random_state=42)
real_names['Label'] = 0

# Combine fictional and real names
df = pd.concat([fictional_names, real_names], ignore_index=True)

# Save balanced reference lists
pd.DataFrame(reference_real, columns=['Name']).to_csv("Balanced_Real_Reference_List.csv", index=False)
pd.DataFrame(reference_fictional, columns=['Name']).to_csv("Balanced_Fictional_Reference_List.csv", index=False)
df.to_csv("MainDataset.csv", index=False)

print(f"✅ Reference lists balanced: {len(reference_real)} real, {len(reference_fictional)} fictional.")
print(f"📌 Final dataset size: {len(df)}")
