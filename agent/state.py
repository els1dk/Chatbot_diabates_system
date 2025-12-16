class State:
    def __init__(self, glucose, risk):
        self.glucose = glucose
        self.risk = risk

    def is_goal(self):
        return self.glucose == "normal"

    def __eq__(self, other):
        return self.glucose == other.glucose and self.risk == other.risk

    def __hash__(self):
        return hash((self.glucose, self.risk))
