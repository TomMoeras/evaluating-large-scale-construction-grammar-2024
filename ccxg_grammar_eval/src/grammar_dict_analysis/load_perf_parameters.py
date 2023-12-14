import pickle
from pathlib import Path
from typing import Dict, Tuple
import sys
# Add the 'src' directory to sys.path
src_path = Path(__file__).resolve().parent.parent  # nopep8
sys.path.append(str(src_path))  # nopep8
from grammar_dict_analysis.performance import *
from grammar_corpus import Grammar, GrammarManager
from grammar_dict_analysis.load_grammar_dict_pickle import load_grammar_dict_from_pickle, GrammarDictUnpickler


def main():
    grammar_dict_pickle_dir = Path(
        __file__).parent.parent.parent / "grammar_dict_pickle"
    pickle_file_path = grammar_dict_pickle_dir / \
        "exp_full_train_4000_predictions_grammar_objects.pkl"

    # Load grammar_dict and corpus from the pickle file
    grammar_dict, corpus = load_grammar_dict_from_pickle(pickle_file_path)
    print("Loaded grammar_dict and corpus from pickle file")

    manager = GrammarManager(corpus)
    manager.grammar_dict = grammar_dict
    print("Created GrammarManager")

    performance_params = PerformanceParameters(grammar_dict)

    PerformanceParameters.save_to_pickle(
        performance_params, "performance_params.pkl")

    sentence_frame_info_manager = SentenceFrameInfoManager.from_performance_object(
        performance_params, manager)
    print("Created SentenceFrameInfoManager")

    SentenceFrameInfoManager.save_to_pickle(
        sentence_frame_info_manager, "sentence_frame_info_manager.pkl")


if __name__ == "__main__":
    main()
