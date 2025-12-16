from agent.search import a_star


class DiabetesAgent:
    def __init__(self, state):
        self.state = state

    def plan(self):
        return a_star(self.state)
