# Step 0: Install necessary packages
# pip install pandas numpy tensorflow scikit-learn arabic-reshaper python-bidi

import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ================================
# Step 1: Load data
# ================================
train_df = pd.read_csv("data/80%_Train.csv")
val_df   = pd.read_csv("data/10%_Val.csv")
test_df  = pd.read_csv("data/10%_test.csv")

# Map sentiment to 0/1
for df in [train_df, val_df, test_df]:
    df['sentiment'] = df['sentiment'].map({'negative':0, 'positive':1})

# ================================
# Step 2: Preprocess text
# ================================
def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^ء-ي\s]", "", text)  # Keep only Arabic letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

for df in [train_df, val_df, test_df]:
    df['review'] = df['review'].apply(normalize_arabic)

# ================================
# Step 3: Tokenize & Pad sequences
# ================================
MAX_WORDS = 5000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_df['review'])  # Fit only on training data

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['review']), maxlen=MAX_LEN)
X_val   = pad_sequences(tokenizer.texts_to_sequences(val_df['review']), maxlen=MAX_LEN)
X_test  = pad_sequences(tokenizer.texts_to_sequences(test_df['review']), maxlen=MAX_LEN)

y_train = np.array(train_df['sentiment'])
y_val   = np.array(val_df['sentiment'])
y_test  = np.array(test_df['sentiment'])

# ================================
# Step 5: Build LSTM Model
# ================================
model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ================================
# Step 6: Train Model
# ================================
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# ================================
# Step 7: Evaluate Model
# ================================
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ================================
# Step 8: Save Model
# ================================
model.save("moroccan_sentiment_lstm.h5")
