import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load cleaned dataset
df = pd.read_csv("data/processed/hinglish_sentiment_cleaned.csv")

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 1. Sentiment Distribution
# --------------------------------------------------

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

plt.figure(figsize=(7, 5))
df["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Comment Length
# --------------------------------------------------

df["text_length"] = df["text"].str.len()

print("\nAverage comment length by sentiment:")
print(df.groupby("sentiment")["text_length"].mean())

plt.figure(figsize=(8, 5))

df.boxplot(
    column="text_length",
    by="sentiment"
)

plt.title("Comment Length by Sentiment")
plt.suptitle("")
plt.xlabel("Sentiment")
plt.ylabel("Characters")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Most Common Words
# --------------------------------------------------

all_words = " ".join(df["text"]).split()

word_counts = Counter(all_words)

print("\nTop 20 most common words:")

for word, count in word_counts.most_common(20):
    print(word, ":", count)


# --------------------------------------------------
# 4. Sample Comments
# --------------------------------------------------

print("\nPositive examples:")
print(df[df["sentiment"] == "Positive"]["text"].head(5).to_string(index=False))

print("\nNegative examples:")
print(df[df["sentiment"] == "Negative"]["text"].head(5).to_string(index=False))

print("\nNeutral examples:")
print(df[df["sentiment"] == "Neutral"]["text"].head(5).to_string(index=False))