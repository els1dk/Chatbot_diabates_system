import random

GENERAL_INFO_RESPONSES = [
    "Diabetes is a condition where blood sugar levels are too high.",
    "Diabetes happens when the body cannot properly control blood sugar.",
    "Diabetes affects how the body uses glucose for energy."
]

def generate_response(intent, plan):
    if intent == "reduce_glucose":
        return "To reduce your glucose level, you should: " + ", ".join(plan)

    if intent == "diet_advice":
        return "Diet advice: follow a balanced, low-sugar diet."

    if intent == "exercise_advice":
        return "Exercise advice: regular moderate activity helps control sugar."

    if intent == "general_info":
        return random.choice(GENERAL_INFO_RESPONSES)

    return "I'm here to help with diabetes-related questions."
