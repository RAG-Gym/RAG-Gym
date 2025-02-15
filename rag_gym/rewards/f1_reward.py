from rag_gym import State, Action
from rag_gym.rewards.utils import f1_score

class F1Reward:
    
    def __init__(self):
        pass

    def __call__(self, state: State, actions: Action | list[Action], next_states: None | State | list[State] = None):
        assert state.truth is not None
        return_list = True if type(actions) == list else False
        if type(actions) == Action:
            actions = [actions]
        rewards = []
        for action in actions:
            if action.query is not None:
                rewards.append(0)
            elif action.answer is None:
                rewards.append(0)
            else:
                rewards.append(f1_score(action.answer, state.truth)[0])
        if not return_list:
            return rewards[0]
        return rewards