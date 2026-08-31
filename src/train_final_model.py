import pandas as pd
import joblib

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load improved dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/hinglish_sentiment_improved.csv"
)

X = df["text"]
y = df["sentiment"]

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. Word TF-IDF
# --------------------------------------------------

word_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)


# --------------------------------------------------
# 4. Character TF-IDF
# --------------------------------------------------

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=15000
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)


# --------------------------------------------------
# 5. Combine features
# --------------------------------------------------

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

print("Combined TF-IDF shape:", X_train_combined.shape)


# --------------------------------------------------
# 6. Train final model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train_combined,
    y_train
)


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test_combined)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nFinal Model Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 8. Save model and vectorizers
# --------------------------------------------------

joblib.dump(
    word_vectorizer,
    "models/word_tfidf.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/char_tfidf.pkl"
)

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)


print("\n----------------------------------")
print("FINAL MODEL SAVED SUCCESSFULLY!")
print("----------------------------------")

print("models/word_tfidf.pkl")
print("models/char_tfidf.pkl")
print("models/sentiment_model.pkl")
