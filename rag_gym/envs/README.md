# `rag_gym/envs`

The information-retrieval environment that agents interact with. RAG-Gym frames
agentic search as a high-level Markov Decision Process (MDP): the agent observes
a `State` (question plus information-seeking history) and takes a macro-action
(`Action`) that is either a search query or a final answer. The environment
follows a Gym-style `make` / `reset` / `step` interface.

## Files

| File | Contents |
|------|----------|
| `env.py` | `Env` class and the `make(**kwargs)` factory. `reset(question, truth=None)` initializes the state; `step(action)` executes a query (retrieving documents and appending them to the history) or terminates the episode on an answer, returning `(observation, reward, terminated, truncated, info)`. |
| `state.py` | `State` and `History`. The state holds the original question and the accumulated `(query, retrieved documents)` history; `History` supports JSON serialization for prompting. |
| `action.py` | `Action`, the macro-action wrapper distinguishing query actions from answer actions. |
| `IR.py` | Retrieval backend. Defines the supported corpora and retrievers, builds/loads BM25 and dense indexes, downloads corpora from Hugging Face on first use, and fuses rankings with Reciprocal Rank Fusion (RRF). |
| `utils.py` | `RetrievalSystemCached`, a caching wrapper over the retriever to avoid recomputing repeated queries. |

## Environment configuration

`rag_gym.make` accepts the retrieval settings used throughout the paper:

```python
env = rag_gym.make(
    retriever_name="RRF-BGE",     # lexical + dense fusion
    corpus_name="Wikipedia",
    max_iter=10,                  # maximum search steps per question
    k=32,                         # documents retrieved per query
    rrf_k=60,                     # RRF constant
    cache=False,
    HNSW=True,
)
```

### Supported corpora (`corpus_name`)

| Name | Sources |
|------|---------|
| `Wikipedia` | English Wikipedia |
| `Wikipedia_HotpotQA` | HotpotQA Wikipedia dump (`RAG-Gym/wiki_hotpotqa`) |
| `MedText` | Medical textbooks + StatPearls |
| `MedCorp` | PubMed + textbooks + StatPearls + Wikipedia |

### Supported retrievers (`retriever_name`)

| Name | Components |
|------|-----------|
| `BM25` | Lexical only |
| `BGE` | Dense (`BAAI/bge-base-en-v1.5`) |
| `MedCPT` | Dense (`ncbi/MedCPT-Query-Encoder`) |
| `RRF-BGE` | BM25 + BGE (general-domain default) |
| `RRF-2` | BM25 + MedCPT (medical default) |
| `RRF-4` | BM25 + Contriever + SPECTER + MedCPT |

Retrieval hyperparameters are held fixed across agents so that performance
differences reflect agent behavior rather than the retrieval system. `Git-lfs`
and `Java/21` (for BM25) are required; corpora and indexes are downloaded to
`./corpus` on first use.
