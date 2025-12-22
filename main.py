
import numpy as np
from models.diabetes_nn import train_diabetes_model, get_risk_level
from models.intent_nn import train_intent_model, predict_intent
from models.simulator import apply_scenario
from agent.state import State
from agent.agent import DiabetesAgent
from chatbot import generate_response


# ==================== CONFIGURATION ====================

CONFIDENCE_THRESHOLD = 0.20
DIABETES_DATA_PATH = "data/diabetes.csv"
INTENTS_DATA_PATH = "data/intents.csv"
DEBUG_MODE = False  # Set to True to see intent predictions


# ==================== HELPER FUNCTIONS ====================

def extract_glucose_state(text):
    """
    Extract glucose state from user input text.
    
    Args:
        text: User input string
        
    Returns:
        str: 'high', 'low', or 'normal'
    """
    text = text.lower()
    if "low" in text:
        return "low"
    elif "high" in text or "reduce" in text:
        return "high"
    else:
        return "normal"


def get_user_input():
    """
    Collect patient health metrics interactively.
    
    Returns:
        numpy.ndarray: Patient data array with 8 features
    """
    print("\n" + "="*60)
    print("DIABETES RISK ASSESSMENT - Patient Data Input")
    print("="*60)
    
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
        print("❌ Invalid input. Please enter numeric values.")
        return get_user_input()


def detect_simulation_scenario(user_input):
    """
    Detect which simulation scenario the user is asking about.
    
    Args:
        user_input: User's question text
        
    Returns:
        str or None: Scenario name or None if not detected
    """
    user_lower = user_input.lower()
    
    # Check for negative keywords
    has_negative = any(word in user_lower for word in 
                      ["don't", "dont", "no", "stop", "not", "avoid", "bad", "poor", "junk"])
    
    # Exercise scenarios
    if "walk" in user_lower or "exercise" in user_lower:
        return "no_exercise" if has_negative else "walk_daily"
    
    # Diet scenarios - check negative first
    elif "diet" in user_lower or "eat" in user_lower or "food" in user_lower:
        # Negative diet keywords
        if has_negative or "junk" in user_lower or "unhealthy" in user_lower:
            return "poor_diet"
        # Positive diet keywords
        elif "healthy" in user_lower or "good" in user_lower or "strict" in user_lower or "better" in user_lower:
            return "healthy_diet"
        # Default to poor diet if just "eat" with negative context
        elif has_negative:
            return "poor_diet"
    
    # Stress scenario
    elif "stress" in user_lower:
        return "reduce_stress"
    
    return None


def run_simulation(patient, risk, scenario_name, diabetes_model, scaler):
    """
    Run a what-if simulation and generate response.
    
    Args:
        patient: Patient data array
        risk: Current risk level
        scenario_name: Name of scenario to simulate
        diabetes_model: Trained diabetes prediction model
        scaler: Data scaler for normalization
        
    Returns:
        str: Formatted simulation response
    """
    modified_patient, description = apply_scenario(patient, scenario_name)
    new_risk = get_risk_level(diabetes_model, scaler, modified_patient)
    
    # Get probabilities
    current_prob = diabetes_model.predict(scaler.transform(patient))[0][0]
    new_prob = diabetes_model.predict(scaler.transform(modified_patient))[0][0]
    
    # Build response
    response = (f"🔮 Simulating: {description}\n\n"
               f"📊 Current Risk: {risk.upper()} (probability: {current_prob:.2f})\n"
               f"📊 Predicted Risk: {new_risk.upper()} (probability: {new_prob:.2f})\n\n")
    
    # Add interpretation
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
        prob_diff = current_prob - new_prob
        if prob_diff > 0.01:
            response += f"📉 Positive trend! Risk probability decreased by {prob_diff:.2f}"
        elif prob_diff < -0.01:
            response += f"📈 Warning! Risk probability increased by {abs(prob_diff):.2f}"
        else:
            response += "➡️ Your risk would remain similar, but this is still a healthy choice!"
    
    return response


# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point."""
    
    print("\n" + "="*60)
    print("DIABETES CHATBOT - Intelligent Health Management System")
    print("="*60)
    
    # Train models
    print("\n[1/2] Training diabetes risk prediction model...")
    diabetes_model, scaler, diabetes_history = train_diabetes_model(DIABETES_DATA_PATH)
    
    print("\n[2/2] Training intent classification model...")
    intent_model, vectorizer, label_encoder, intent_history = train_intent_model(INTENTS_DATA_PATH)
    
    print("\n✅ Models trained successfully!")
    
    # Generate training visualization plots (optional)
    try:
        from models.visualize import plot_both_models
        plot_both_models(intent_history, diabetes_history)
    except ImportError:
        print("\n⚠️  Matplotlib not installed. Skipping visualization plots.")
        print("   To generate plots, install matplotlib: pip install matplotlib")
    
    # Get patient data
    patient = get_user_input()
    risk = get_risk_level(diabetes_model, scaler, patient)
    
    # Get actual probability for debugging
    prob = diabetes_model.predict(scaler.transform(patient))[0][0]
    
    print(f"\n{'='*60}")
    print(f"RISK ASSESSMENT RESULT: {risk.upper()}")
    print(f"Diabetes Probability: {prob:.4f}")
    print(f"{'='*60}")
    
    # Chat loop
    print("\n💬 Chat with the bot! (Type 'exit' to quit)")
    print("Examples:")
    print("  • 'Give me diet advice'")
    print("  • 'What if I walk daily?'")
    print("  • 'Help me plan my day'")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == "exit":
            print("\n👋 Bot: Goodbye! Stay healthy!")
            break
        
        # Predict intent
        intent, confidence = predict_intent(intent_model, vectorizer, label_encoder, user_input)
        
        # Debug output (optional)
        if DEBUG_MODE:
            print(f"[DEBUG] Predicted: {intent} (confidence: {confidence:.3f})")
        
        # Handle "what if" questions and simulation keywords - bypass confidence check
        simulation_keywords = ["what if", "simulate", "show me", "predict", "compare"]
        if any(keyword in user_input.lower() for keyword in simulation_keywords):
            intent = "simulate"
        elif confidence < CONFIDENCE_THRESHOLD:
            if DEBUG_MODE:
                print(f"[DEBUG] Confidence {confidence:.3f} < {CONFIDENCE_THRESHOLD}, using fallback")
            intent = "fallback"
        
        # Extract glucose state for agent
        glucose_state = extract_glucose_state(user_input)
        state = State(glucose_state, risk)
        
        # Plan actions using agent
        agent = DiabetesAgent(state)
        plan = agent.plan()
        
        # Handle simulation requests
        if intent == "simulate" or "what if" in user_input.lower():
            scenario = detect_simulation_scenario(user_input)
            if scenario:
                response = run_simulation(patient, risk, scenario, diabetes_model, scaler)
            else:
                response = generate_response(intent, plan, risk)
        else:
            response = generate_response(intent, plan, risk)
        
        print(f"\nBot: {response}")


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()
