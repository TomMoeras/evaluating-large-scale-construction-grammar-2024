import csv
import re
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import List
from pathlib import Path
from tqdm import tqdm
from grammar_corpus import (
    Corpus,
    Frame,
    VerbAtlasFrames,
    Grammar,
    GrammarManager,
    Prediction,
)


def load_macro_evaluation_data(file_path: str, manager: GrammarManager):
    df = pd.read_csv(file_path)

    # Remove rows containing NaN values
    df = df.dropna()

    # Filter out rows with invalid grammar_ids
    # df = df[df["grammar_id"].str.isnumeric()]

    # Convert the "grammar_id" column to integer
    df["grammar_id"] = df["grammar_id"].astype(int)

    for _, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Loading macro evaluation data",
        unit="grammars",
    ):
        grammar_id = row["grammar_id"]
        grammar = manager.get_grammar(grammar_id)
        grammar.set_evaluation_scores(
            row["precision"],
            row["recall"],
            row["f1_score"],
            row["nr_of_correct_predictions"],
            row["nr_of_predictions"],
            row["nr_of_gold_standard_predictions"],
        )


def load_grammar_data(file_path: str, manager: GrammarManager):
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row

        for row in tqdm(csvreader, desc="Loading grammar data", unit="grammars"):
            grammar_id = int(row[0])
            grammar_size = int(row[1])
            manager.grammar_size_dict[grammar_id] = grammar_size

            heuristics = None
            if row[2]:
                heuristics = row[2].strip("()").replace("-", "_").lower().split()

            learning_modes = None
            if row[3]:
                learning_modes_raw = (
                    row[3].strip("()").replace("-", "_").lower().split()
                )
                learning_modes = []
                for mode in learning_modes_raw:
                    if mode.startswith("argm"):
                        learning_modes.append("argm_group")
                    else:
                        learning_modes.append(mode)
                learning_modes = list(set(learning_modes))  # Remove duplicates

            excluded_rolesets = None
            if row[4]:
                excluded_rolesets_raw = (
                    row[4].strip("()").replace("-", "_").lower().split()
                )
                if "nil" not in excluded_rolesets_raw:
                    excluded_rolesets = ["aux"]
                else:
                    excluded_rolesets = excluded_rolesets_raw

            # Create the config_str
            config_values = [
                (heuristics, "heuristics"),
                (learning_modes, "learning_modes"),
                (excluded_rolesets, "excluded_rolesets"),
            ]

            config_str = "; ".join(
                f"{label}: {', '.join(map(str, value))}"
                for value, label in config_values
                if value is not None
            )

            # Create a new Grammar object and add it to the manager
            new_grammar = Grammar(grammar_id, config_str, grammar_size)
            manager.add_grammar(new_grammar)

            # No need to pass the grammar_size parameter
            manager.get_grammar(grammar_id)
            # manager.add_data_to_grammar(
            # grammar_id, heuristics, learning_modes, excluded_rolesets)


def load_frame_prediction_data(file_path: str, manager: GrammarManager, corpus: Corpus):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["grammar_id"])
    df = df[df["support"] != 0]
    df = df.replace(".*NIL.*", 0, regex=True)
    df["f1_score"] = pd.to_numeric(df["f1_score"])
    df = df.sort_values(by=["f1_score", "support"], ascending=False)

    for _, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Loading frame prediction data",
        unit="frames",
    ):
        grammar_id = row["grammar_id"]
        grammar = manager.get_grammar(grammar_id)

        # Get the corresponding VerbAtlas frame information
        mapping = corpus.verb_atlas_frames.get_verbatlas_mapping(row["frame_name"])
        va_frame_id = mapping["va_frame_id"]
        va_frame_name = mapping["va_frame_name"]

        grammar.add_frame_roleset_performance(
            row["frame_name"],
            row["precision"],
            row["recall"],
            row["f1_score"],
            row["support"],
            va_frame_name,
            va_frame_id,
        )


def _parse_frame_roles(
    frame_roles: str, verb_atlas_frames: VerbAtlasFrames
) -> List[Frame]:
    frame_data = []
    frame_pattern = r"([^:]+): (.+?)(?=;;|$)"
    role_pattern = r"(\w+) \[(.+?)\]: (.+?)(?= //|$)"

    for frame_match in re.finditer(frame_pattern, frame_roles):
        frame_name, roles_str = frame_match.groups()
        roles = []

        for role_match in re.finditer(role_pattern, roles_str):
            role, indices, string = role_match.groups()
            indices = list(map(int, indices.split(" | ")))
            string = string.replace("~", ",")
            roles.append({"role": role, "indices": indices, "string": string})

        frame_name = frame_name.replace(";; ", "")
        lemma_name = frame_name.split(".")[0].lower()
        roles.append({"role": "FEE", "indices": [0], "string": lemma_name})

        # Add the VerbAtlas frame information
        mapping = verb_atlas_frames.get_verbatlas_mapping(frame_name)
        va_frame_id = mapping["va_frame_id"]
        va_frame_name = mapping["va_frame_name"]
        role_mappings = mapping["role_mappings"]

        # Convert the roles to va_roles using the role_mappings
        va_roles = []
        for role in roles:
            converted_role = verb_atlas_frames._convert_propbank_role(role["role"])
            va_role = role_mappings.get(converted_role, "unknown")
            va_roles.append(
                {"role": va_role, "indices": role["indices"], "string": role["string"]}
            )

        frame = Frame(
            frame_name, lemma_name, roles, va_roles, va_frame_id, va_frame_name
        )

        frame_data.append(frame)

    return frame_data


def load_micro_evaluation_data(file_path: str, manager: GrammarManager, corpus: Corpus):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

    df.replace("~", ",", inplace=True, regex=True)

    df = df.dropna()

    # Replace 'NIL' with 0 only in rows where recall is 0
    df.loc[df["recall"] == 0, :] = df.loc[df["recall"] == 0, :].replace(
        ".*NIL.*", 0, regex=True
    )
    df = df.replace(".*NIL.*", np.nan, regex=True)

    # Remove rows where the number of frames is zero
    df = df[df["frames"] != 0]

    # remove rows where the number of nr_of_gold_standard_predictions is zero
    # df = df[df["nr_of_gold_standard_predictions"] != 0]

    # Remove brackets from the 'source_file' and 'elapsed_time' columns
    df["source_file"] = df["source_file"].replace(r"\(|\)", "", regex=True)
    df["elapsed_time"] = df["elapsed_time"].replace(r"\(|\)", "", regex=True)

    # convert all the values to numeric values except the source_file column
    numeric_columns = df.columns.drop("source_file")
    numeric_columns = numeric_columns.drop("frames_info")
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    df["frames_and_roles"] = df["frames_info"].astype(str)

    for _, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Loading micro evaluation data",
        unit="sentences",
    ):
        grammar_id = row["prediction_nr"]
        source_file_index = manager.corpus._source_file_to_index(row["source_file"])
        unique_sentence_id = (source_file_index, row["sentence_id"])
        grammar = manager.get_grammar(grammar_id)
        sentence = corpus.get_sentence(unique_sentence_id)
        # Parse frames and roles and add them to the Grammar object
        frame_roles = _parse_frame_roles(
            row["frames_and_roles"], corpus.verb_atlas_frames
        )
        grammar.frame_role_data[unique_sentence_id] = frame_roles
        # Create a new SentenceEvaluation object and add it to the Grammar object
        prediction = Prediction(sentence, row.to_dict(), frame_roles)

        # if sentence is not None:
        #     same_lemma_count, same_verb_lemma_count = prediction.count_same_lemmas()
        # else:
        #     same_lemma_count, same_verb_lemma_count = 0, 0

        grammar.add_prediction(unique_sentence_id, prediction)
        grammar.micro_evaluation_info.append(
            {
                "sen_id": unique_sentence_id,
                "nr_tokens": row["tokens"],
                "nr_frames": row["frames"],
                "nr_aux": row["aux_frames"],
                "nr_roles": row["roles"],
                "nr_core": row["core_roles"],
                "nr_argm": row["non_core_roles"],
                # "same_lemma": same_lemma_count,
                # "same_verb": same_verb_lemma_count,
                "time": row["elapsed_time"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
                "nr_of_correct_predictions": row["nr_of_correct_predictions"],
                "nr_of_predictions": row["nr_of_predictions"],
                "nr_of_gold_standard_predictions": row[
                    "nr_of_gold_standard_predictions"
                ],
            }
        )


def save_grammar_to_pickle(manager: GrammarManager, file_path: str):
    data = {
        "grammar_dict": manager.grammar_dict,
        "corpus": manager.corpus,  # Add this line
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def main():
    # Define the data directory path
    data_dir = Path(__file__).parent.parent / "data"
    grammar_corpus_prediction_evaluation_data_dir = (
        data_dir / "grammar_corpus_prediction_evaluation_data"
    )
    verbatlas_data_dir = data_dir / "verbatlas_data"
    grammar_dict_pickle_dir = Path(__file__).parent.parent / "grammar_dict_pickle"

    # Load corpus data
    corpus_file_path = grammar_corpus_prediction_evaluation_data_dir / "corpus_data.csv"

    # corpus_file_path = grammar_corpus_prediction_evaluation_data_dir / \
    #     "full_dev" / "corpus_full_dev.csv"

    pb2va_file_path = verbatlas_data_dir / "pb2va.tsv"
    VA_frame_info_file_path = verbatlas_data_dir / "VA_frame_info.tsv"
    verb_atlas_frames = VerbAtlasFrames(pb2va_file_path, VA_frame_info_file_path)
    corpus = Corpus(corpus_file_path, verb_atlas_frames=verb_atlas_frames)

    # Pass the Corpus instance to the GrammarManager constructor
    manager = GrammarManager(corpus)

    # Replace with the path to your grammar data file
    grammar_file_path = (
        grammar_corpus_prediction_evaluation_data_dir / "grammar_data.csv"
    )
    # Replace with the path to your evaluation data file
    macro_evaluation_file_path = (
        grammar_corpus_prediction_evaluation_data_dir / "macro_evaluation_data.csv"
    )
    # Replace with the path to your frame prediction data file
    frame_prediction_file_path = (
        grammar_corpus_prediction_evaluation_data_dir / "frame_prediction_data.csv"
    )
    # Replace with the path to your time_info data file
    micro_evaluation_file_path = (
        grammar_corpus_prediction_evaluation_data_dir / "micro_evaluation_data.csv"
    )

    # # Replace with the path to your grammar data file
    # grammar_file_path = grammar_corpus_prediction_evaluation_data_dir / \
    #     "full_dev" / "grammar_data_full_dev.csv"
    # # Replace with the path to your evaluation data file
    # macro_evaluation_file_path = grammar_corpus_prediction_evaluation_data_dir / "full_dev" / \
    #     "macro_evaluation_data_full_dev.csv"
    # # Replace with the path to your frame prediction data file
    # frame_prediction_file_path = grammar_corpus_prediction_evaluation_data_dir / "full_dev" / \
    #     "frame_prediction_data_full_dev.csv"
    # # Replace with the path to your time_info data file
    # micro_evaluation_file_path = grammar_corpus_prediction_evaluation_data_dir / "full_dev" / \
    #     "micro_evaluation_data_full_dev.csv"

    load_grammar_data(grammar_file_path, manager)
    load_macro_evaluation_data(macro_evaluation_file_path, manager)
    load_frame_prediction_data(frame_prediction_file_path, manager, corpus)
    load_micro_evaluation_data(micro_evaluation_file_path, manager, corpus)

    # # Access and use grammar objects by grammar_id
    # for grammar_id in manager.grammar_dict.keys():
    #     grammar = manager.get_grammar(grammar_id)
    #     # print(f"Grammar with grammar_id {grammar_id}: {grammar}")
    #     # print(f"Config: {grammar.config}")
    #     # print(f"Macro evaluation scores: {grammar.evaluation_scores}")
    #     # print(f"Frame predictions: {grammar.frame_roleset_performance}")
    #     # print(f"Micro evaluation scores: {grammar.micro_evaluation_info}")
    #     # print(f"Frame role data: {grammar.frame_role_data}\n")  # Added line

    # set pickle file name to timestamp
    pickle_file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_grammar_dict.pkl"
    # Save grammar objects to a pickle file
    pickle_file_path = grammar_dict_pickle_dir / pickle_file_name
    save_grammar_to_pickle(manager, pickle_file_path)
    print("Saved grammar_dict to pickle file in " + str(pickle_file_path))


if __name__ == "__main__":
    main()
