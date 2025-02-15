from rag_gym import State, Action, RetrievalSystemCached, EMReward

class Env:
    def __init__(
        self, 
        retriever_name: str = None, 
        corpus_name: str = None, 
        max_iter: int = 5, 
        k: int = 32, 
        rrf_k: int = 60, 
        cache = True,
        **kwargs,
    ):
        self.max_iter = max_iter
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.k = k
        self.rrf_k = rrf_k
        self.state = None
        self.curr_iter = 0
        if self.retriever_name is None or self.corpus_name is None:
            self.retrieval_system = None
        else:
            self.retrieval_system = RetrievalSystemCached(retriever_name=self.retriever_name, corpus_name=self.corpus_name, cache=cache, **kwargs)
        self.reward_model = EMReward()

    def info(self):
        return {
            "curr_iter": self.curr_iter,
            "max_iter": self.max_iter,
            "retriever_name": self.retriever_name,
            "corpus_name": self.corpus_name,
            "k": self.k,
            "rrf_k": self.rrf_k,
        }

    def reset(self, question: str, truth: str | None = None):
        self.curr_iter = 0
        self.state = State(question=question, truth=truth, terminated=False, truncated=self.curr_iter>=self.max_iter)        
        return self.state, self.info()

    def close(self):
        import rag_gym
        rag_gym.envs.utils.cached_retrievers.clear()

    def step(self, action: Action):
        # if type(action) == str:
        #     action = Action(query=action)
        next_state = self.transition(self.state, action)
        self.curr_iter += 1
        terminated = self.is_terminal(next_state)
        truncated = self.curr_iter >= self.max_iter
        next_state.terminated = terminated
        next_state.truncated = truncated
        reward = self.get_reward(self.state, action, next_state)
        self.state = next_state
        return next_state, reward, terminated, truncated, self.info()

    def transition(self, state: State, action: Action):
        history = state.history.copy()
        if action.query is None or self.curr_iter + 1 == self.max_iter:
            next_state = State(question=state.question, history=history, truth=state.truth, answer=action.answer)
        else:
            query = action.query
            assert self.retrieval_system is not None
            documents = self.retrieval_system.retrieve(query, k=self.k, rrf_k=self.rrf_k)[0]
            history.add_qd(query=query, documents=documents)
            next_state = State(question=state.question, history=history, truth=state.truth)
        return next_state

    def is_terminal(self, state):
        if state.is_terminal():
            return True
        return False

    def get_reward(self, prev_state, action, curr_state):
        if curr_state.truth is None:
            return None
        return self.reward_model(prev_state, action, curr_state)

def make(**kwargs) -> Env:
    return Env(**kwargs)