import sys  # nopep8
from pathlib import Path  # nopep8
# Add the 'src' directory to sys.path
src_path = Path(__file__).resolve().parent.parent  # nopep8
sys.path.append(str(src_path))  # nopep8
from grammar_corpus import Corpus, Grammar

import pickle
from typing import Dict, Tuple
from pickle import Unpickler


class GrammarDictUnpickler(Unpickler):
    def find_class(self, module, name):
        if module == "corpus_objects":
            module = "grammar_corpus.corpus.corpus_objects"
        return super().find_class(module, name)


def load_grammar_dict_from_pickle(file_path: str) -> Tuple[Dict[int, Grammar], Corpus]:
    with open(file_path, "rb") as f:
        data = GrammarDictUnpickler(f).load()
    # Convert keys to integers
    grammar_dict = {int(k): v for k, v in data["grammar_dict"].items()}
    corpus = data["corpus"]
    return grammar_dict, corpus
