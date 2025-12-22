import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


def train_diabetes_model(csv_path):
    data = pd.read_csv(csv_path)

    # Detect target column automatically
    target_col = None
    for col in data.columns:
        if col.lower() in ["outcome", "target", "diabetes"]:
            target_col = col
            break

    if target_col is None:
        raise ValueError("Target column not found in diabetes dataset")

    X = data.drop(columns=[target_col]).values
    y = data[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    history = model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

    return model, scaler, history


def get_risk_level(model, scaler, patient_data):
    patient_data = scaler.transform(patient_data)
    prob = model.predict(patient_data)[0][0]

    if prob > 0.7:
        return "high"
    elif prob > 0.4:
        return "medium"
    else:
        return "low"
