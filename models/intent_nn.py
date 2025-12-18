import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def train_intent_model(csv_path):
    data = pd.read_csv(csv_path)

    if "text" not in data.columns or "intent" not in data.columns:
        raise ValueError("intents.csv must contain 'text' and 'intent' columns")

    texts = data["text"].astype(str)
    labels = data["intent"].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()

    model = Sequential()
    model.add(Dense(16, input_shape=(X.shape[1],), activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(len(np.unique(y)), activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(X, y, epochs=50, batch_size=8, validation_split=0.2)

    return model, vectorizer, label_encoder


def predict_intent(model, vectorizer, label_encoder, sentence):
    vec = vectorizer.transform([sentence]).toarray()
    prediction = model.predict(vec)
    intent_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return label_encoder.inverse_transform([intent_index])[0], confidence
