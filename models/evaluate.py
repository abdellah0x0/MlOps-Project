import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import re

MAX_LEN = 100
MODEL_PATH = "models/moroccan_sentiment_lstm.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# -------------------------------
# Load model and tokenizer
# -------------------------------
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------------
# Load test data
# -------------------------------
test_df = pd.read_csv("data/10%_test.csv")

# Drop rows with missing review or sentiment
test_df = test_df.dropna(subset=['review', 'sentiment'])

# Map sentiment to 0/1
test_df['sentiment'] = test_df['sentiment'].map({'negative':0, 'positive':1})

# Drop rows that could not be mapped (NaN after mapping)
test_df = test_df.dropna(subset=['sentiment'])

# Ensure sentiment is int
test_df['sentiment'] = test_df['sentiment'].astype(int)

# -------------------------------
# Normalize reviews (same as training)
# -------------------------------
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^ء-ي\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

test_df['review'] = test_df['review'].apply(normalize_arabic)

# Drop rows that became empty after normalization
test_df = test_df[test_df['review'] != ""]

# -------------------------------
# Check if test set is empty
# -------------------------------
if test_df.empty:
    raise ValueError("Test set is empty after cleaning. Check your CSV content and preprocessing!")

# -------------------------------
# Tokenize and pad sequences
# -------------------------------
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['review'].tolist()), maxlen=MAX_LEN)
y_test = np.array(test_df['sentiment'])

# Predict and evaluate
y_pred = (model.predict(X_test, batch_size=64) > 0.5).astype("int32").reshape(-1)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
