from rag_gym.envs.IR import RetrievalSystem, Retriever, DocExtracter, corpus_names, retriever_names

cached_retrievers = dict()
cached_corpora = dict()

class RetrievalSystemCached(RetrievalSystem):
    '''
        add cache to the original RetrievalSystem
    '''
    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", HNSW=False, cache=False):
        global cached_retrievers
        global cached_corpora
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                if f"{retriever}_{corpus}" not in cached_retrievers:
                    cached_retrievers[f"{retriever}_{corpus}"] = Retriever(retriever, corpus, db_dir, HNSW=HNSW)
                self.retrievers[-1].append(cached_retrievers[f"{retriever}_{corpus}"])
        self.cache = cache
        if self.cache:
            if self.corpus_name not in cached_corpora:
                cached_corpora[self.corpus_name] = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
            self.docExt = cached_corpora[self.corpus_name]
        else:
            self.docExt = None