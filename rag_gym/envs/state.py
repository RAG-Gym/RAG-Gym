class History:
    def __init__(self, qd_list: list[dict[str, str]] | None = None):
        if qd_list is None:
            qd_list = []
        for qd in qd_list:
            assert "query" in qd and "documents" in qd, "Each item must have 'query' and 'answer' keys"
        self.qd_list = qd_list
    def __len__(self):
        return len(self.qd_list)
    def __getitem__(self, index):
        if isinstance(index, slice):
            return History(self.qd_list[index])
        elif isinstance(index, int):
            return self.qd_list[index]
    def add_qd(self, query, documents):
        self.qd_list.append({
            "query": query,
            "documents": documents
        })
    def return_as_json(self, return_documents=False):
        if not return_documents:
            return [{"query":item.get("query", None)} for item in self.qd_list]
        return self.qd_list
    def return_as_string(self, return_documents=False):
        if not return_documents:
            return "\n".join([f"Query: {item['query']}" for item in self.qd_list])
        return "\n".join([f"Query: {item['query']}\nRelevant Documents: {str(item['documents'])}" for item in self.qd_list])
    def copy(self):
        return History([qd.copy() for qd in self.qd_list])

class State:
    def __init__(
        self, 
        question: str, 
        history: History | None = None, 
        terminated: bool = False, 
        truncated: bool = False, 
        answer: str | None = None, 
        truth: float | None = None
    ):
        '''
            question: the original question to be answered
            history: the currect information-seeking history
                [
                    {
                        "query": an information-seeking query,
                        "documents": [
                            {
                                "id": id of a relevant document,
                                "title": title of a relevant document,
                                "content": content of a relevant document
                            }
                        ]
                    },
                ]
            terminated: True if the currect state is a terminal state else False
            truncated: True if the maximum iteration number has been reached else False
            answer: the predicted answer
            truth: (training only) the ground truth answer
        '''
        self.question = question
        self.history = history if history is not None else History()
        self.terminated = terminated
        self.truncated = truncated
        self.answer = answer
        self.truth = truth
    def is_terminal(self):
        if self.answer is not None:
            self.terminated = True
        return self.terminated
    def return_as_json(self, return_documents=False):
        return {"question": self.question, "history": self.history.return_as_json(return_documents=return_documents), "answer": self.answer}
    def return_as_string(self, return_documents=False):
        return f"Here is the original question:\n{self.question}\n\nHere is the information-seeking history:\n{self.history.return_as_string(return_documents=return_documents)}\n\nHere is the predicted answer:\n{self.answer}"
    def copy(self):
        return State(
            question = self.question,
            history = self.history.copy(),
            terminated = self.terminated,
            truncated = self.truncated,
            answer = self.answer,
            truth = self.truth,
        )

