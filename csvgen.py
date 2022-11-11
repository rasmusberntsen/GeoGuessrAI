import os
import pandas as pd

df = pd.DataFrame(columns=['country', 'filename'])
map = {
    'de': 'Germany',
    'dk': 'Denmark',
    'ee': 'Estonia',
    'es': 'Spain',
    'fr': 'France',
    'gb': 'United Kingdom',
    'gr': 'Greece',
    'it': 'Italy',
    'no': 'Norway',
    'pl': 'Poland',
    'ro': 'Romania',
    'se': 'Sweden',
    'ua': 'Ukraine',
}

# For each subfolder in images
for folder in os.listdir("images"):
    # For each file in the subfolder
    for file in os.listdir(f"images/{folder}"):
        # Add the country and filename to the dataframe
        filename = f"{folder}/{file}"
        df = df.append({'country': map[folder], 'filename': filename}, ignore_index=True)

# Save the dataframe as a csv with ; as sep
df.to_csv("countries.csv", sep=";")

# pd.read_csv("countries.csv", sep=";")