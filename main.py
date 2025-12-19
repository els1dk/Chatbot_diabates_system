import numpy as np
from models.diabetes_nn import train_diabetes_model, get_risk_level
from models.intent_nn import train_intent_model, predict_intent
from models.simulator import apply_scenario, get_available_scenarios
from agent.state import State
from agent.agent import DiabetesAgent
from chatbot import generate_response


# --------- Helper function ---------
# --------- Helper function ---------
def extract_glucose_state(text):
    text = text.lower()
    if "low" in text:
        return "low"
    elif "high" in text or "reduce" in text:
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
    intent, confidence = predict_intent(
        intent_model, vectorizer, label_encoder, user_input
    )
    
    # Special handling for "what if" questions - bypass confidence check
    if "what if" in user_input.lower():
        intent = "simulate"
    elif confidence < 0.50:  # Lowered threshold from 0.65 to 0.50
        intent = "fallback"

    # 2. Extract glucose state from user text
    glucose_state = extract_glucose_state(user_input)

    # 3. Create agent state (THIS IS THE CONNECTION)
    state = State(glucose_state, risk)

    # 4. Plan actions using agentic search
    agent = DiabetesAgent(state)
    plan = agent.plan()

    # 5. What-If Simulation Mode
    response = None
    if intent == "simulate" or "what if" in user_input.lower():
        # Detect scenario from user input
        user_lower = user_input.lower()
        scenario_detected = None
        
        # Check for negative keywords first
        has_negative = any(word in user_lower for word in ["don't", "dont", "no", "stop", "not", "avoid"])
        
        if "walk" in user_lower or "exercise" in user_lower:
            if has_negative:
                scenario_detected = "no_exercise"
            else:
                scenario_detected = "walk_daily"
        elif "diet" in user_lower or "eat" in user_lower:
            if "healthy" in user_lower or "good" in user_lower or "strict" in user_lower or "better" in user_lower:
                scenario_detected = "healthy_diet"
            elif "junk" in user_lower or "poor" in user_lower or "bad" in user_lower or has_negative:
                scenario_detected = "poor_diet"
        elif "stress" in user_lower:
            scenario_detected = "reduce_stress"
        
        if scenario_detected:
            # Run simulation
            modified_patient, description = apply_scenario(patient, scenario_detected)
            new_risk = get_risk_level(diabetes_model, scaler, modified_patient)
            
            # Get probabilities for debugging
            current_prob = diabetes_model.predict(scaler.transform(patient))[0][0]
            new_prob = diabetes_model.predict(scaler.transform(modified_patient))[0][0]
            
            response = (f"🔮 Simulating: {description}\n\n"
                       f"📊 Current Risk: {risk.upper()} (probability: {current_prob:.2f})\n"
                       f"📊 Predicted Risk: {new_risk.upper()} (probability: {new_prob:.2f})\n\n")
            
            if new_risk != risk:
                if new_risk == "low":
                    response += "✅ Great news! This change could significantly improve your health."
                elif new_risk == "medium" and risk == "high":
                    response += "📈 Positive change! Your risk would decrease."
                elif new_risk == "high" and risk != "high":
                    response += "⚠️ Warning! This could increase your risk."
                else:
                    response += "📊 Your risk level would change."
            else:
                if new_prob < current_prob:
                    response += f"📉 Positive trend! Risk probability decreased by {(current_prob - new_prob):.2f}"
                elif new_prob > current_prob:
                    response += f"📈 Warning! Risk probability increased by {(new_prob - current_prob):.2f}"
                else:
                    response += "➡️ Your risk would remain similar, but this is still a healthy choice!"

    # 6. Generate chatbot response (if not simulation)
    if response is None:
        response = generate_response(intent, plan, risk)
    
    print("Bot:", response)
