import json
import os
import pandas as pd

class KIQA:

    def __init__(self, data, split="test", dir="./dataset"):
        assert data in ["medqa", "hotpotqa", "2wikimultihopqa", "bamboogle"]
        if split == "test":
            if data == "hotpotqa":
                self.dataset = pd.DataFrame(json.load(open(os.path.join(dir, "hotpotqa/hotpot_dev_fullwiki_v1.json")))).iloc[-1000:,[0,2,1]]
                self.dataset.reset_index(inplace=True, drop=True)
                self.dataset.columns = ["index", "question", "answer"]
            elif data == "2wikimultihopqa":
                self.dataset = pd.DataFrame(json.load(open(os.path.join(dir, "2wikimultihopqa/dev.json")))).iloc[-1000:,[0,2,7]]
                self.dataset.reset_index(inplace=True, drop=True)
                self.dataset.columns = ["index", "question", "answer"]
            elif data == "bamboogle":
                self.dataset = pd.read_csv(os.path.join(dir, "bamboogle/Bamboogle_Prerelease.tsv"), sep='\t').iloc[:,[0,4]]
                self.dataset.reset_index(inplace=True)
                self.dataset.columns = ["index", "question", "answer"]
            elif data == "medqa":
                self.dataset = pd.read_json(os.path.join(dir, "medqa/phrases_no_exclude_test.jsonl"), lines=True).iloc[:,[0,2,4]]
                self.dataset.reset_index(inplace=True)
                self.dataset.columns = ["index", "question", "options", "answer"]
        else:
            if data == "medqa":
                self.dataset = pd.read_csv(os.path.join(dir, "medqa_train_1000.csv"))
            elif data == "hotpotqa":
                self.dataset = pd.read_csv(os.path.join(dir, "hotpotqa_dev_1000.csv"))
                self.dataset.columns = ["index", "question", "answer"]
        self.index = self.dataset["index"].values

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.dataset.iloc[key].to_dict()
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")