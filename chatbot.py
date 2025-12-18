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
            return f"{risk_prefix}Strictly low-carb, high-fiber diet is crucial. Avoid sugary drinks and processed foods."
        elif risk == "medium":
            return f"{risk_prefix}Watch your carb intake and choose whole grains suitable for pre-diabetes."
        else:
            return f"{risk_prefix}Great work! Maintain a balanced diet with plenty of vegetables to stay healthy."

    if intent == "exercise_advice":
        if risk == "high":
            return f"{risk_prefix}Start with light walking and monitor signals. Consult your doctor first."
        elif risk == "medium":
            return f"{risk_prefix}Aim for 150 minutes of moderate activity per week to lower your risk."
        else:
            return f"{risk_prefix}Keep it up! Regular activity is key to maintaining your low risk."

    if intent == "daily_plan":
        if risk == "high":
            return (f"{risk_prefix}Here is a safe daily plan:\n"
                    "- Morning: Check glucose, light breakfast (oatmeal).\n"
                    "- Mid-day: Short walk, salad with lean protein.\n"
                    "- Evening: Grilled veggies/fish, check glucose before bed.")
        elif risk == "medium":
            return (f"{risk_prefix}Suggested Routine:\n"
                    "- Morning: Balanced breakfast.\n"
                    "- Afternoon: 30 min brisk walk.\n"
                    "- Evening: Avoid late night carbs.")
        else:
            return (f"{risk_prefix}Healthy Routine:\n"
                    "- Maintain your regular healthy meals.\n"
                    "- Keep active with your favorite sports/hobbies.")

    if intent == "general_info":
        return random.choice(GENERAL_INFO_RESPONSES)

    if intent == "fallback":
        return "I'm not sure I understand. I can help with diet, exercise, or a daily plan. Could you please rephrase?"

    return "I'm here to help. You can ask about diet, exercise, or a daily plan."
