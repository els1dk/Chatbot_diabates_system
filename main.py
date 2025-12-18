import numpy as np
from models.diabetes_nn import train_diabetes_model, get_risk_level
from models.intent_nn import train_intent_model, predict_intent
from agent.state import State
from agent.agent import DiabetesAgent
from chatbot import generate_response


# --------- Helper function ---------
# --------- Helper function ---------
def extract_glucose_state(text):
    text = text.lower()
    if "low" in text:
        return "low"
    elif "high" in text:
        return "high"
    else:
        return "normal"


def get_user_input():
    print("\nPlease enter the following details for diabetes risk assessment:")
    try:
        pregnancies = float(input("Pregnancies: "))
        glucose = float(input("Glucose: "))
        blood_pressure = float(input("BloodPressure: "))
        skin_thickness = float(input("SkinThickness: "))
        insulin = float(input("Insulin: "))
        bmi = float(input("BMI: "))
        dpf = float(input("DiabetesPedigreeFunction: "))
        age = float(input("Age: "))
        
        return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return get_user_input()


# --------- Train models ---------
print("Training diabetes risk model...")
diabetes_model, scaler = train_diabetes_model(
    "data/diabetes.csv"
)

print("Training intent classification model...")
intent_model, vectorizer, label_encoder = train_intent_model(
    "data/intents.csv"
)

# --------- Patient profile (from user input) ---------
patient = get_user_input()
risk = get_risk_level(diabetes_model, scaler, patient)

print("\nPredicted risk level from Kaggle-trained NN:", risk)


# --------- Chat loop ---------
print(f"\nExample: 'I want to reduce my glucose' or 'Give me diet advice'")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    # 1. Predict intent
    intent = predict_intent(
        intent_model, vectorizer, label_encoder, user_input
    )

    # 2. Extract glucose state from user text
    glucose_state = extract_glucose_state(user_input)

    # 3. Create agent state (THIS IS THE CONNECTION)
    state = State(glucose_state, risk)

    # 4. Plan actions using agentic search
    agent = DiabetesAgent(state)
    plan = agent.plan()

    # 5. Generate chatbot response
    response = generate_response(intent, plan, risk)
    print("Bot:", response)
