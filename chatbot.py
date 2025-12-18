import random

GENERAL_INFO_RESPONSES = [
    "Diabetes is a condition where blood sugar levels are too high.",
    "Diabetes happens when the body cannot properly control blood sugar.",
    "Diabetes affects how the body uses glucose for energy."
]

def generate_response(intent, plan, risk):
    risk_prefix = ""
    if risk == "high":
        risk_prefix = "[ALERT: HIGH RISK] Consult a specialist. "
    elif risk == "medium":
        risk_prefix = "[NOTE: MEDIUM RISK] Monitor closely. "
    else:
        risk_prefix = "[LOW RISK] Good job! "

    if intent == "reduce_glucose":
        return f"{risk_prefix}To reduce your glucose level, consider this plan: " + ", ".join(plan)

    if intent == "diet_advice":
        if risk == "high":
            return f"{risk_prefix}Strictly low-carb, high-fiber diet is crucial."
        return f"{risk_prefix}Diet advice: follow a balanced, low-sugar diet."

    if intent == "exercise_advice":
        if risk == "high":
            return f"{risk_prefix}Start with light walking and monitor signals."
        return f"{risk_prefix}Exercise advice: regular moderate activity helps control sugar."

    if intent == "general_info":
        return random.choice(GENERAL_INFO_RESPONSES)

    return "I'm here to help with diabetes-related questions."
