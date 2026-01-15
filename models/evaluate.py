import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

MAX_LEN = 100
MODEL_PATH = "models/moroccan_sentiment_lstm.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"
TEST_DATA_PATH = "data/10%_Test.csv"

# load model and tokenizer
print("Loading model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# normalization function
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

# load data
print("Loading test data...")
df = pd.read_csv(TEST_DATA_PATH)
print(f"Loaded {len(df)} samples")

print("\n=== DATA INSPECTION ===")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nSentiment column unique values: {df['sentiment'].unique()}")
print(f"Sentiment value counts:\n{df['sentiment'].value_counts()}")

# clean sentiment labels - convert whatever format to 0/1
df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
df = df.dropna(subset=['sentiment', 'review'])
df['sentiment'] = df['sentiment'].astype(int)

print(f"\nAfter converting to numeric: {len(df)} samples")
print(f"Label distribution: {df['sentiment'].value_counts().to_dict()}")

# normalize reviews
df['review'] = df['review'].apply(normalize_arabic)
df = df[df['review'].str.len() > 0].reset_index(drop=True)

print(f"After normalization: {len(df)} samples")

# tokenize
sequences = tokenizer.texts_to_sequences(df['review'].tolist())
valid_idx = [i for i, s in enumerate(sequences) if len(s) > 0]
df = df.iloc[valid_idx].reset_index(drop=True)
sequences = [sequences[i] for i in valid_idx]

print(f"After tokenization: {len(sequences)} samples")

# prepare data
X = pad_sequences(sequences, maxlen=MAX_LEN)
y_true = df['sentiment'].values

print(f"\nFinal dataset: {len(y_true)} samples")
print(f"Class distribution: {np.bincount(y_true)}")

# predict
print("\nMaking predictions...")
y_prob = model.predict(X, batch_size=64, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Samples:    {len(y_true)}")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print("="*50)

# classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], digits=4))

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.4f} | F1: {f1:.4f}")
plt.tight_layout()
plt.savefig("confusion_matrix_lstm.png", dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved to: confusion_matrix_lstm.png")
plt.show()

print("\nEvaluation complete!")
