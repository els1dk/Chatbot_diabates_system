"""
Agent Performance Evaluation Module

This module implements quantitative evaluation of the diabetes agent's 
decision-making performance using the PEAS framework.

PEAS Framework Definition:
- P (Performance Measure): Correctness of recommended actions, ability to reach 
  goal state (normal glucose), alignment with patient risk level
- E (Environment): Simulated patient health environment with states derived 
  from Kaggle dataset features
- A (Actuators): Chatbot responses providing lifestyle recommendations
- S (Sensors): User text input and patient health attributes (numerical data)
"""

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from agent.state import State
from agent.agent import DiabetesAgent
import numpy as np


# Ground truth: Expected actions for different patient states
# Based on medical best practices and agent design
GROUND_TRUTH = {
    ("high", "high"): "avoid_sugar",
    ("high", "medium"): "walk_30_minutes",
    ("high", "low"): "walk_30_minutes",
    ("low", "high"): "eat_healthy_meal",
    ("low", "medium"): "eat_healthy_meal",
    ("low", "low"): "monitor_glucose",
    ("normal", "high"): "monitor_glucose",
    ("normal", "medium"): "monitor_glucose",
    ("normal", "low"): "monitor_glucose",
}


def create_test_cases():
    """
    Create test cases covering all possible state combinations.
    Returns a list of test case dictionaries.
    """
    test_cases = []
    
    for (glucose, risk), expected_action in GROUND_TRUTH.items():
        test_cases.append({
            "state": State(glucose, risk),
            "expected_action": expected_action,
            "description": f"Glucose: {glucose}, Risk: {risk}"
        })
    
    return test_cases


def evaluate_agent(test_cases, agent_class):
    """
    Evaluate agent performance using accuracy and F1 score.
    
    Args:
        test_cases: List of test case dictionaries
        agent_class: Agent class to evaluate (DiabetesAgent)
    
    Returns:
        dict: Evaluation metrics including accuracy, F1 score, and detailed report
    """
    y_true = []
    y_pred = []
    results = []
    
    for case in test_cases:
        state = case["state"]
        expected = case["expected_action"]
        
        # Get agent's planned actions
        agent = agent_class(state)
        plan = agent.plan()
        
        # Extract first action (primary recommendation)
        predicted = plan[0] if plan else "none"
        
        y_true.append(expected)
        y_pred.append(predicted)
        
        results.append({
            "description": case["description"],
            "expected": expected,
            "predicted": predicted,
            "correct": expected == predicted
        })
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get unique labels for F1 calculation
    labels = list(set(y_true + y_pred))
    f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, zero_division=0)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "labels": labels,
        "detailed_results": results,
        "total_cases": len(test_cases),
        "correct_predictions": sum(1 for r in results if r["correct"])
    }


def print_evaluation_results(metrics):
    """
    Print formatted evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*70)
    print("AGENT PERFORMANCE EVALUATION (PEAS Framework)")
    print("="*70)
    
    print("\n📊 OVERALL METRICS:")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   Correct Predictions: {metrics['correct_predictions']}/{metrics['total_cases']}")
    
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print(metrics['classification_report'])
    
    print("\n🔍 CONFUSION MATRIX:")
    print(f"   Labels: {metrics['labels']}")
    print(metrics['confusion_matrix'])
    
    print("\n📝 DETAILED RESULTS:")
    for i, result in enumerate(metrics['detailed_results'], 1):
        status = "✅" if result['correct'] else "❌"
        print(f"   {status} Test {i}: {result['description']}")
        print(f"      Expected: {result['expected']}, Predicted: {result['predicted']}")
    
    print("\n" + "="*70)


def run_evaluation():
    """
    Main evaluation function. Creates test cases and evaluates agent.
    """
    print("Creating test cases...")
    test_cases = create_test_cases()
    
    print(f"Running evaluation on {len(test_cases)} test cases...")
    metrics = evaluate_agent(test_cases, DiabetesAgent)
    
    print_evaluation_results(metrics)
    
    return metrics


if __name__ == "__main__":
    # Run evaluation when script is executed directly
    run_evaluation()
