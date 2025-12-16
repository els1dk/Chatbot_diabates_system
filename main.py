import numpy as np
from models.diabetes_nn import train_diabetes_model, get_risk_level
from models.intent_nn import train_intent_model, predict_intent
from agent.state import State
from agent.agent import DiabetesAgent
from chatbot import generate_response


# Train models
diabetes_model, scaler = train_diabetes_model("data/diabetes.csv")
intent_model, vectorizer, label_encoder = train_intent_model("data/intents.csv")

# Example patient (must match dataset feature count)
patient = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
risk = get_risk_level(diabetes_model, scaler, patient)

print("Predicted risk level:", risk)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    intent = predict_intent(intent_model, vectorizer, label_encoder, user_input)

    state = State("high", risk)
    agent = DiabetesAgent(state)
    plan = agent.plan()

    response = generate_response(intent, plan)
    print("Bot:", response)
