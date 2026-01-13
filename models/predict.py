import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

MAX_LEN = 100

# Load model & tokenizer
model = load_model("models/moroccan_sentiment_lstm.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load raw reviews
raw_df = pd.read_csv("data/raw_reviews.csv")

# Normalize reviews the same way as training
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

raw_df['review'] = raw_df['review'].apply(normalize_arabic)

# Tokenize & pad all reviews at once
X = pad_sequences(tokenizer.texts_to_sequences(raw_df['review'].tolist()), maxlen=MAX_LEN)

# Predict in batch
y_pred = (model.predict(X, batch_size=64) > 0.5).astype("int32")
raw_df['sentiment_pred'] = ["positive" if p==1 else "negative" for p in y_pred]

raw_df.to_csv("data/raw_predicted.csv", index=False)
print("Predictions done! Check raw_predicted.csv")
