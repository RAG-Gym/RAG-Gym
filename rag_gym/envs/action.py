
class Action:
    '''
        An action should be a tuple of (query, answer, action_str)
    '''
    def __init__(self, query: str | None = None, answer: str | None = None, action_string: str | None = None):
        self.query = query
        self.answer = answer
        self.action_string = action_string
    def return_as_json(self):
        return {
            "query": self.query,
            "answer": self.answer
        }
    def return_as_string(self):
        if self.action_string:
            return self.action_string
        return f"Query: {self.query} | Answer: {self.answer}"