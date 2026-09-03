# 💬 Hinglish Sentiment Analysis

A Machine Learning and Natural Language Processing (NLP) project that analyzes **Hinglish comments** written in Roman script and classifies them into three sentiment categories:

* 😊 Positive
* 😞 Negative
* 😐 Neutral

The project uses **Word-level TF-IDF** and **Character-level TF-IDF** features combined with **Logistic Regression** to handle spelling variations commonly found in Hinglish text.

---

## 📌 Project Overview

Hinglish is a combination of Hindi and English commonly used in social media, online discussions, and YouTube comments.

The same Hinglish word can be written in different ways, for example:

```text
accha
acha
achha
```

A word-based model may treat these as different words.

To handle these variations, this project combines:

**Word TF-IDF + Character TF-IDF**

The combined features are then passed to a **Logistic Regression** classifier to predict the sentiment.

---

## 🔄 Project Workflow

```text
              Hinglish Comment
                     ↓
              Text Preprocessing
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
   Word TF-IDF              Character TF-IDF
        │                         │
        └────────────┬────────────┘
                     ↓
          Combined TF-IDF Features
                     ↓
            Logistic Regression
                     ↓
                 Sentiment
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
   Positive       Negative       Neutral
```

---

## 📊 Dataset

The project uses a Hinglish sentiment dataset containing comments labeled as Positive, Negative, or Neutral.

### Dataset Size

* **Total comments:** 3,181
* **Training samples:** 2,544
* **Testing samples:** 637

### Sentiment Distribution

| Sentiment |   Samples |
| --------- | --------: |
| Negative  |     1,421 |
| Positive  |       992 |
| Neutral   |       768 |
| **Total** | **3,181** |

The dataset is split using an **80/20 stratified train-test split**.

---

## 🧹 Data Preprocessing

The raw dataset is processed and cleaned before model training.

### Raw Dataset

```text
data/raw/hinglish.sentiment.csv
```

### Final Processed Dataset

```text
data/processed/hinglish_sentiment_improved.csv
```

The final model uses the improved processed dataset.

---

## 🔤 Feature Engineering

### Word-level TF-IDF

Word TF-IDF captures important words and word combinations.

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

This creates up to **10,000 word-level features** using unigrams and bigrams.

### Character-level TF-IDF

Character TF-IDF captures character-level patterns and helps handle spelling variations.

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=15000
)
```

This creates up to **15,000 character-level features**.

### Combined Features

The two feature sets are combined using sparse matrices:

```python
X_train_combined = hstack([
    X_train_word,
    X_train_char
])
```

The final feature representation contains **25,000 features**.

---

## 🤖 Machine Learning Model

The final classification model is **Logistic Regression**.

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
```

`class_weight="balanced"` helps compensate for the imbalance between the sentiment classes.

---

## 📈 Model Performance

The final model was evaluated on **637 unseen test samples**.

### Accuracy

**66.09%**

### Classification Report

| Class                | Precision |   Recall | F1-Score |
| -------------------- | --------: | -------: | -------: |
| Negative             |      0.69 |     0.74 |     0.71 |
| Neutral              |      0.57 |     0.64 |     0.61 |
| Positive             |      0.71 |     0.56 |     0.62 |
| **Macro Average**    |  **0.66** | **0.65** | **0.65** |
| **Weighted Average** |  **0.67** | **0.66** | **0.66** |

The model achieved **66.09% test accuracy** and a **0.65 macro F1-score**.

---

## 🔮 Prediction

The trained model and TF-IDF vectorizers are stored in the `models/` directory:

```text
models/
├── char_tfidf.pkl
├── sentiment_model.pkl
└── word_tfidf.pkl
```

The `predict.py` script loads these saved files and predicts the sentiment of new Hinglish comments.

### Run Prediction

From the project root:

```bash
python src/predict.py
```

Example:

```text
Enter a Hinglish comment: movie bahut acchi hai

Predicted Sentiment: Positive
```

You can enter comments such as:

```text
movie bahut acchi hai
```

```text
movie bilkul bakwas thi
```

```text
movie theek thi
```

The model predicts:

```text
Positive
Negative
Neutral
```

Type `exit` to stop the program.

---

## 📁 Project Structure

```text
Hinglish-Sentiment-Analysis/
│
├── data/
│   ├── raw/
│   │   └── hinglish.sentiment.csv
│   │
│   └── processed/
│       └── hinglish_sentiment_improved.csv
│
├── models/
│   ├── char_tfidf.pkl
│   ├── sentiment_model.pkl
│   └── word_tfidf.pkl
│
├── src/
│   ├── eda.py
│   ├── preprocess_data.py
│   ├── train_final_model.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data processing
* **NumPy** – Numerical operations
* **Scikit-learn** – TF-IDF and Logistic Regression
* **SciPy** – Sparse matrix operations
* **Joblib** – Model saving and loading
* **Matplotlib** – Data visualization
* **Seaborn** – Data visualization

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vidhim06/Hinglish-Sentiment-Analysis.git
```

### 2. Navigate to the project

```bash
cd Hinglish-Sentiment-Analysis
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

Run the final training script:

```bash
python src/train_final_model.py
```

The script:

1. Loads the processed dataset
2. Splits the dataset into training and testing sets
3. Creates Word TF-IDF features
4. Creates Character TF-IDF features
5. Combines both feature sets
6. Trains Logistic Regression
7. Evaluates the model
8. Saves the trained model and vectorizers

---

## 🔮 Future Improvements

* Build a Flask-based web application
* Improve performance using larger Hinglish datasets
* Experiment with multilingual transformer models
* Add prediction confidence scores
* Deploy the application online

---

## 👩‍💻 Author

**Vidhi Mittal**

B.Tech – Computer Science & Engineering
Specialization in Data Science
