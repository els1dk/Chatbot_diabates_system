from agent.state import State


def get_actions(state):
    actions = []

    if state.risk == "high":
        actions.append(("walk", 2))
        actions.append(("avoid_sugar", 1))
        actions.append(("check_glucose", 1))

    if state.risk == "low":
        actions.append(("light_exercise", 1))

    return actions


def apply_action(state, action):
    name, cost = action

    if name == "walk":
        return State("normal", state.risk), cost
    if name == "avoid_sugar":
        return State("normal", state.risk), cost
    if name == "drink_water":
        return state, cost
