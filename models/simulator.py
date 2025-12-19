import numpy as np

# Scenario modifiers for what-if simulations
# Each scenario modifies patient features based on lifestyle changes

SCENARIOS = {
    "walk_daily": {
        "description": "Walking 30 minutes daily",
        "modifiers": {
            "Glucose": -30,     # Major glucose reduction
            "BMI": -3.0,        # Significant BMI reduction
            "BloodPressure": -10 # Major blood pressure improvement
        }
    },
    "no_exercise": {
        "description": "No regular exercise",
        "modifiers": {
            "Glucose": +35,     # Major glucose increase
            "BMI": +3.5,        # Significant BMI increase
            "BloodPressure": +12 # Major blood pressure increase
        }
    },
    "healthy_diet": {
        "description": "Following a low-carb, high-fiber diet",
        "modifiers": {
            "Glucose": -40,     # Dramatic glucose reduction
            "BMI": -4.0,        # Major weight loss
            "Insulin": -100     # Dramatic insulin improvement
        }
    },
    "poor_diet": {
        "description": "High sugar and processed food diet",
        "modifiers": {
            "Glucose": +45,     # Dramatic glucose increase
            "BMI": +4.5,        # Major weight gain
            "Insulin": +150     # Dramatic insulin worsening
        }
    },
    "reduce_stress": {
        "description": "Managing stress levels",
        "modifiers": {
            "Glucose": -15,
            "BloodPressure": -15
        }
    }
}

# Feature indices in the patient array
FEATURE_MAP = {
    "Pregnancies": 0,
    "Glucose": 1,
    "BloodPressure": 2,
    "SkinThickness": 3,
    "Insulin": 4,
    "BMI": 5,
    "DiabetesPedigreeFunction": 6,
    "Age": 7
}


def apply_scenario(patient_data, scenario_name):
    """
    Apply a what-if scenario to patient data.
    
    Args:
        patient_data: numpy array of patient features
        scenario_name: name of scenario from SCENARIOS dict
        
    Returns:
        modified_patient: numpy array with scenario applied
        description: human-readable description of scenario
    """
    if scenario_name not in SCENARIOS:
        return patient_data.copy(), "Unknown scenario"
    
    scenario = SCENARIOS[scenario_name]
    modified = patient_data.copy()
    
    # Apply modifiers
    for feature, change in scenario["modifiers"].items():
        if feature in FEATURE_MAP:
            idx = FEATURE_MAP[feature]
            modified[0][idx] = max(0, modified[0][idx] + change)  # Ensure non-negative
    
    return modified, scenario["description"]


def get_available_scenarios():
    """Return list of available scenario names and descriptions."""
    return [(name, info["description"]) for name, info in SCENARIOS.items()]
