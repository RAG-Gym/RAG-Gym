# `rag_gym/agents`

Agent architectures evaluated in RAG-Gym. Every agent subclasses `BaseAgent`
(`base_agent.py`) and implements a common interface, so they can be swapped
without changing the environment or the training and inference pipelines:

- `generate_action(state, temperature, num_actions)` returns one or more
  candidate `Action` objects for the current `State`.
- `post_process(action_str)` parses a raw model completion into an `Action`.
- `apply_template(state, ...)` renders the prompt shown to the language model.
- `score(state, actions, ...)` ranks candidate actions with the process reward
  model when critic-guided inference is enabled.

## Files

| File | Agent | Description |
|------|-------|-------------|
| `base_agent.py` | `BaseAgent` | Shared base class. Loads the actor LLM through `LLMEngine` and, when `reward_llm_name` is set, loads and merges the LoRA process reward model for critic-guided (Best-of-N) inference. |
| `direct.py` | `DirectAgent` | Answers in a single step from parametric memory, with no reasoning or retrieval. |
| `cot.py` | `CoTAgent` | Generates a chain-of-thought before answering, still in a single pass without retrieval. |
| `rag.py` | `RAGAgent` | Issues the question as one query, retrieves once, then answers over the retrieved context. |
| `react.py` | `ReActAgent` | Interleaves reasoning and actions across multiple steps, deciding at each step whether to search or answer. |
| `search_o1.py` | `Searcho1Agent` | Extends ReAct with a document-summarization module (`RAGModule`) that condenses retrieved documents into query-specific answers before further reasoning. |
| `research.py` | `ReSearchAgent` | Our Re$^2$Search agent. Adds the Reasoning Reflection module: it drafts a step-by-step answer, identifies the first unverified claim, and turns that claim into the next targeted query. Re$^2$Search++ is this agent combined with DPO training and critic-guided inference. |
| `utils.py` | `LLMEngine` | Unified generation backend for local (vLLM/transformers) and API models. |

## Functional module matrix

Each architecture instantiates a subset of the six functional modules from the
paper (Fig. 2b). Re$^2$Search is the only agent that instantiates the full
stack, with Reasoning Reflection as its distinguishing component.

| Agent | Answer Gen. | Question Reasoning | Retrieval Aug. | Query Gen. | Doc. Summ. | Reasoning Reflection |
|-------|:-----------:|:------------------:|:--------------:|:----------:|:----------:|:--------------------:|
| Direct        | ✓ |   |   |   |   |   |
| CoT           | ✓ | ✓ |   |   |   |   |
| RAG           | ✓ |   | ✓ |   |   |   |
| ReAct         | ✓ | ✓ | ✓ | ✓ |   |   |
| Search-o1     | ✓ | ✓ | ✓ | ✓ | ✓ |   |
| Re$^2$Search  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Usage

Agents are instantiated through the top-level package (see the repository
`README.md` for full inference examples):

```python
import rag_gym
agent = rag_gym.ReSearchAgent(
    llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api=False,
    reward_llm_name=None,                 # set to a PRM checkpoint for Best-of-N
    rag_llm_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
)
```

The available agent classes are `DirectAgent`, `CoTAgent`, `RAGAgent`,
`ReActAgent`, `Searcho1Agent`, and `ReSearchAgent`. In the command-line
pipelines (`src/`, `rag_gym/algorithms/`) they are selected with
`--agent_type {direct,cot,rag,react,search_o1,research}`.
