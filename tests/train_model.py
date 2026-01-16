import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
from tensorflow.keras.callbacks import EarlyStopping
import os

MAX_WORDS = 5000
MAX_LEN = 100
BATCH_SIZE = 64
EPOCHS = 10
MODEL_PATH = "models/moroccan_sentiment_lstm.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

os.makedirs("models", exist_ok=True)

# Load processed train & validation data
train_df = pd.read_csv("data/80%_Train.csv")
val_df = pd.read_csv("data/10%_Val.csv")

# Map sentiment labels to 0/1
for df in [train_df, val_df]:
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

X_train_texts = train_df["review"].tolist()
X_val_texts = val_df["review"].tolist()
y_train = np.array(train_df["sentiment"])
y_val = np.array(val_df["sentiment"])

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_texts), maxlen=MAX_LEN)
X_val = pad_sequences(tokenizer.texts_to_sequences(X_val_texts), maxlen=MAX_LEN)

# Save tokenizer
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train model
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
)

# Save model

model.save(MODEL_PATH)
print(f"Training complete. Model saved to {MODEL_PATH}")
