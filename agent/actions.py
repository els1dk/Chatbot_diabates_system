from agent.state import State


def get_actions(state):
    actions = []

    if state.glucose == "high":
        actions.append(("avoid_sugar", 1))
        actions.append(("walk_30_minutes", 2))

    if state.glucose == "low":
        actions.append(("eat_healthy_meal", 1))

    return actions


def apply_action(state, action):
    name, cost = action

    if name in ["avoid_sugar", "walk_30_minutes"]:
        return State("normal", state.risk), cost

    if name == "eat_healthy_meal":
        return State("normal", state.risk), cost

    return state, cost
