import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100
MODEL_PATH = "models/moroccan_sentiment_lstm.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# normalization
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

# prediction function
def predict_review(review_text):
    review_text = normalize_arabic(review_text)
    seq = tokenizer.texts_to_sequences([review_text])
    if len(seq[0]) == 0:
        return "negative", 0.50   
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(padded, verbose=0)[0][0]
    if np.isnan(prob):
        return "negative", 0.50
    if prob >= 0.5:
        return "positive", float(prob)
    else:
        return "negative", float(1 - prob)

if __name__ == "__main__":
    text = "هذا الفيلم رائع بزاف"
    sentiment, confidence = predict_review(text)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.3f}")
