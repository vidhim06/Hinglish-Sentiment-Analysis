import pandas as pd
import re

# Load dataset
df = pd.read_csv("data/raw/hinglish_sentiment.csv")

print("Original shape:", df.shape)

# Keep only required columns
df = df[["comment", "sentiment"]]

# Rename comment column
df = df.rename(columns={"comment": "text"})

# Remove missing values
df = df.dropna()

# Remove duplicate comments
df = df.drop_duplicates(subset="text")

# Text cleaning function
def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
df["text"] = df["text"].apply(clean_text)

# Remove empty text
df = df[df["text"].str.len() > 0]

print("Cleaned shape:", df.shape)

print("\nFirst 10 cleaned comments:")
print(df.head(10))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# Save processed dataset
df.to_csv(
    "data/processed/hinglish_sentiment_cleaned.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")