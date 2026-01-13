import re
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 5000
MAX_LEN = 100

def normalize_arabic(text):
    """Normalize Arabic/Darija text"""
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^ء-ي\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_file(file_path, save_processed=False):
    """Read CSV, normalize reviews, return dataframe"""
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(normalize_arabic)
    if save_processed:
        df.to_csv(file_path.replace(".csv", "_processed.csv"), index=False)
    return df

def tokenize_and_pad(texts, tokenizer=None, fit=False):
    """Tokenize and pad sequences"""
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_WORDS)
    if fit:
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded, tokenizer

def save_tokenizer(tokenizer, path="models/tokenizer.pkl"):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path="models/tokenizer.pkl"):
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer
