import heapq
from agent.actions import get_actions, apply_action


def heuristic(state):
    """
    Simple heuristic:
    cost is higher if glucose is high
    """
    if state.glucose == "high":
        return 1
    return 0


def a_star(start_state):
    # Priority queue: (priority, state, path)
    frontier = []
    heapq.heappush(frontier, (0, start_state, []))

    explored = set()

    while frontier:
        cost, state, path = heapq.heappop(frontier)

        # Goal test
        if state.is_goal():
            return path

        explored.add(state)

        for action in get_actions(state):
            next_state, action_cost = apply_action(state, action)

            if next_state not in explored:
                new_cost = cost + action_cost
                priority = new_cost + heuristic(next_state)
                new_path = path + [action[0]]

                heapq.heappush(
                    frontier,
                    (priority, next_state, new_path)
                )

    return []
