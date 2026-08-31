
import os
import joblib

from scipy.sparse import hstack


# ==================================================
# 1. MODEL PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

WORD_TFIDF_PATH = os.path.join(
    BASE_DIR,
    "models",
    "word_tfidf.pkl"
)

CHAR_TFIDF_PATH = os.path.join(
    BASE_DIR,
    "models",
    "char_tfidf.pkl"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sentiment_model.pkl"
)


# ==================================================
# 2. LOAD TRAINED MODEL AND VECTORIZERS
# ==================================================

word_vectorizer = joblib.load(
    WORD_TFIDF_PATH
)

char_vectorizer = joblib.load(
    CHAR_TFIDF_PATH
)

model = joblib.load(
    MODEL_PATH
)

print("Model and TF-IDF vectorizers loaded successfully!")


# ==================================================
# 3. PREDICTION FUNCTION
# ==================================================

def predict_sentiment(text):

    # Word TF-IDF
    word_features = word_vectorizer.transform(
        [text]
    )

    # Character TF-IDF
    char_features = char_vectorizer.transform(
        [text]
    )

    # Combine Word + Character TF-IDF
    combined_features = hstack([
        word_features,
        char_features
    ])

    # Logistic Regression prediction
    prediction = model.predict(
        combined_features
    )[0]

    return prediction


# ==================================================
# 4. MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("       HINGLISH SENTIMENT ANALYZER")
    print("=" * 50)

    print("\nEnter 'exit' to stop the program.")

    while True:

        text = input(
            "\nEnter a Hinglish comment: "
        )

        # Exit
        if text.lower().strip() == "exit":

            print(
                "\nThank you for using "
                "Hinglish Sentiment Analyzer!"
            )

            break

        # Empty input
        if not text.strip():

            print(
                "Please enter a comment."
            )

            continue

        # Predict
        sentiment = predict_sentiment(
            text
        )

        print(
            "\nPredicted Sentiment:",
            sentiment
        )

