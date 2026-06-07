import pandas as pd

dominance = pd.read_csv("data/processed/features_dominance.csv")
arousal = pd.read_csv("data/processed/features_arousal.csv")
valence = pd.read_csv("data/processed/features_valence.csv")

print(f"Null values in dominance: {dominance.isnull().sum().sum()}")
print(f"Null values in arousal: {arousal.isnull().sum().sum()}")
print(f"Null values in valence: {valence.isnull().sum().sum()}")

print(f"Dominance values: {dominance["label"].value_counts()}")
print(f"Arousal values: {arousal["label"].value_counts()}")
print(f"Valence values: {valence["label"].value_counts()}")