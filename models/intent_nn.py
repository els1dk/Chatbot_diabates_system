import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def train_intent_model(csv_path):
    data = pd.read_csv(csv_path)

    if "text" not in data.columns or "intent" not in data.columns:
        raise ValueError("intents.csv must contain 'text' and 'intent' columns")

    texts = data["text"].astype(str)
    labels = data["intent"].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Improved TF-IDF with better parameters
    vectorizer = TfidfVectorizer(
        max_features=500,  # Reduced features for small dataset
        stop_words="english",
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=1,
        max_df=0.9
    )
    X = vectorizer.fit_transform(texts).toarray()

    # Stratified split to ensure balanced validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Improved model architecture
    model = Sequential()
    model.add(Dense(32, input_shape=(X.shape[1],), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(np.unique(y)), activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train with more epochs but early stopping
    history = model.fit(X_train, y_train, 
              epochs=50, 
              batch_size=8, 
              validation_data=(X_val, y_val), 
              callbacks=[early_stop],
              verbose=1)

    return model, vectorizer, label_encoder, history


def predict_intent(model, vectorizer, label_encoder, sentence):
    vec = vectorizer.transform([sentence]).toarray()
    prediction = model.predict(vec)
    intent_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return label_encoder.inverse_transform([intent_index])[0], confidence
