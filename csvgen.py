import os
import pandas as pd

df = pd.DataFrame(columns=['country', 'filename'])
map = {
    'de': 0,#'Germany',
    'dk': 1,#'Denmark',
    'ee': 2,#'Estonia',
    'es': 3,#'Spain',
    'fr': 4,#'France',
    'gb': 5,#'United Kingdom',
    'gr': 6,#'Greece',
    'it': 7,#'Italy',
    'no': 8,#'Norway',
    'pl': 9,#'Poland',
    'ro': 10,#'Romania',
    'se': 11,#'Sweden',
    'ua': 12,#'Ukraine',
}

# For each subfolder in images
for folder in os.listdir("images"):
    # For each file in the subfolder
    for file in os.listdir(f"images/{folder}"):
        # Add the country and filename to the dataframe
        filename = f"{folder}/{file}"
        df = df.append({'country': map[folder], 'filename': filename}, ignore_index=True)

# Save the dataframe as a csv with ; as sep
df.to_csv("countries.csv", sep=";", index=False)

# pd.read_csv("countries.csv", sep=";")