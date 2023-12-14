import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from IPython.display import display, HTML
from tabulate import tabulate
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import re
import random
from grammar_corpus.corpus import Corpus, Frame, Sentence, VerbAtlasFrames


class Prediction:
    """
    Class for managing grammar predictions of frames for a given sentence.

    Attributes:
    sentence (Sentence): The sentence to predict frames for.
    micro_evaluation_data (Dict[str, any]): Micro-evaluation data for the prediction. Defaults to an empty dictionary.
    grammar_predicted_frames (List[Frame]): Frames predicted by the grammar. Defaults to an empty list.

    Methods:
    visualize_prediction(visualization_type: str) -> Dict[str, str]:
        Visualize the predicted frames in a table and/or using the displacy visualizer from SpaCy.
    _visualize_frames(frames: List[Dict[str, any]], visualization_type: str) -> None:
        Visualize a list of frames in a table and/or using the displacy visualizer from SpaCy.
    _visualize_table(frames: List[Frame]) -> str:
        Visualize a list of frames in an HTML table.
    _visualize_displacy(frames: List[Frame]) -> str:
        Visualize a list of frames using the displacy visualizer from SpaCy.

    The class provides functionality for visualizing the predicted frames in different formats.
    """

    def __init__(
        self,
        sentence: Sentence,
        micro_evaluation_data: Optional[Dict[str, any]] = None,
        grammar_predicted_frames: Optional[List[Frame]] = None,
    ):
        self.sentence = sentence
        self.micro_evaluation_data = micro_evaluation_data or {}
        self.grammar_predicted_frames = grammar_predicted_frames or []

    def __repr__(self):
        return (
            f"sentence={self.sentence}, "
            f"micro_evaluation_data={self.micro_evaluation_data}, "
            f"grammar_predicted_frames={self.grammar_predicted_frames})"
        )

    def visualize_prediction(self, visualization_type: str = "both") -> Dict[str, str]:
        table_html = ""
        displacy_html = ""

        no_prediction_message = "No grammar predictions."

        if visualization_type in ("table", "both"):
            table_html_original = self._visualize_table(self.sentence.frames)
            if self.grammar_predicted_frames:
                table_html_predicted = self._visualize_table(
                    self.grammar_predicted_frames
                )
            else:
                table_html_predicted = no_prediction_message

            table_html = f"<div class='original-frames'><h2>Original Sentence Frames:</h2><div class='table-wrapper'>{table_html_original}</div></div><div class='grammar-frames'><h2>Grammar Predicted Frames:</h2><div class='table-wrapper'>{table_html_predicted}</div></div>"

        if visualization_type in ("displacy", "both"):
            displacy_html_original = self._visualize_displacy(self.sentence.frames)
            if self.grammar_predicted_frames:
                displacy_html_predicted = self._visualize_displacy(
                    self.grammar_predicted_frames
                )
            else:
                displacy_html_predicted = no_prediction_message

            displacy_html = f"<div class='original-frames'><h2>Original Sentence Frames:</h2><div class='displacy-wrapper'>{displacy_html_original}</div></div><div class='grammar-frames'><h2>Grammar Predicted Frames:</h2><div class='displacy-wrapper'>{displacy_html_predicted}</div></div>"
        return {
            "table_html": table_html,
            "displacy_html": displacy_html,
        }

    def _visualize_frames(
        self, frames: List[Dict[str, any]], visualization_type: str
    ) -> None:
        if not frames:
            print("No frames.")
            return

        if visualization_type == "table":
            self._visualize_table(frames)
        elif visualization_type == "displacy":
            self._visualize_displacy(frames)
        else:
            raise ValueError(
                "Invalid visualization_type. Choose 'table' or 'displacy'."
            )

    def _visualize_table(self, frames: List[Frame]) -> str:
        if not frames:
            return "No frames in the input."

        print(f"Sentence: {self.sentence.sentence_string}\n")

        table_html = ""
        for frame in frames:
            table_headers = [f"ID: {self.sentence.sentence_id}", frame.frame_name]
            # Combine the propbank_roles and va_roles with a '/', unless the va_role is 'unknown'
            table_data = [
                [
                    f"{role['role']} ({va_role['role']})"
                    if va_role["role"] != "unknown"
                    else role["role"],
                    role["string"],
                ]
                for role, va_role in zip(frame.roles, frame.va_roles)
            ]

            html_table = tabulate(table_data, headers=table_headers, tablefmt="html")
            table_html += f"{html_table}<br>"

        return table_html

    def _visualize_displacy(self, frames: List[Frame]) -> str:
        if not frames:
            return "No frames in the input."
        sentence = self.sentence

        # Replace `` with " to consider it as one token
        sentence_string = sentence.sentence_string.replace("``", '"')

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence_string)

        table_header = (
            f"<h3>{len(frames)} frames for sentence ID: {sentence.sentence_id}</h3>"
        )

        html = table_header

        for frame in frames:
            entities = []

            for role, va_role in zip(frame.roles, frame.va_roles):
                if role["role"] == "FEE":
                    for token in doc:
                        if token.lemma_ == role["string"]:
                            role_start = token.idx
                            role_end = token.idx + len(token.text)
                            break
                else:
                    # Get the indices of the role string in the doc based on the role indices
                    role_start = doc[role["indices"][0]].idx
                    role_end = doc[role["indices"][-1]].idx + len(
                        doc[role["indices"][-1]]
                    )

                # Combine the propbank_roles and va_roles with a '/', unless the va_role is 'unknown'
                label = (
                    f"{role['role']} ({va_role['role']})"
                    if va_role["role"] != "unknown"
                    else role["role"]
                )

                entities.append(
                    {
                        "start": role_start,
                        "end": role_end,
                        "label": label,
                    }
                )

            render_data = [
                {
                    "text": doc.text,
                    "ents": entities,
                    "title": f"ID: {sentence.sentence_id} {frame.frame_name}",
                }
            ]

            color_mapping = {
                "ARG0": "#FCA311",
                "ARG1": "#2EC4B6",
                "ARG2": "#E63946",
                "ARG3": "#DD6E42",
                "ARG4": "#4EA8DE",
                "FEE": "#57A773",
                "V": "#57A773",
            }

            colors = {}
            for role, va_role in zip(frame.roles, frame.va_roles):
                if va_role["role"] != "unknown":
                    label = f"{role['role']} ({va_role['role']})"
                    # Fall back to the default color if not specified
                    color = color_mapping.get(role["role"], "#EA8189")
                    colors[label] = color
                else:
                    label = role["role"]
                    # Fall back to the default color if not specified
                    color = color_mapping.get(role["role"], "#EA8189")
                    colors[label] = color

            displacy_options = {
                "compact": True,
                "offset_x": 100,
                "distance": 100,
                "manual": True,
                "fine_grained": True,
                "colors": colors,
            }

            frame_html = displacy.render(
                render_data,
                style="ent",
                manual=True,
                options=displacy_options,
                page=True,
                jupyter=False,
                minify=True,
            )
            html += frame_html

        return html


@dataclass
class Grammar:
    """
    Class for managing and manipulating grammar information.

    Attributes:
    grammar_id (int): Unique identifier for the grammar.
    grammar_size (int): Size of the grammar.
    config (Dict[str, Optional[List[str]]]): Configurations used for the grammar including heuristics, learning modes, and excluded rolesets.
    evaluation_scores (Dict[str, float]): Evaluation scores for the grammar. Contains precision, recall, and f1 scores.
    frame_roleset_performance (Dict[str, Dict[str, float]]): Performance of different frame rolesets.
    micro_evaluation_info (List[Dict[str, any]]): Micro-evaluation information for the grammar.

    Methods:
    get_summary(sentence_id: Tuple[int, int], include_evaluation_scores: bool, include_micro_evaluation_info: bool) -> Dict[str, Any]:
        Generate a summary of the grammar's information and its prediction for a sentence.
    _parse_config(config: str) -> Dict[str, Optional[List[str]]]:
        Parse a configuration string into a dictionary.
    add_prediction(sen_id: Tuple[int, int], predictions: Prediction):
        Add a prediction to the grammar's prediction dictionary.
    get_prediction(sen_id: Tuple[int, int]) -> Optional[Prediction]:
        Retrieve a prediction from the grammar's prediction dictionary.
    add_data(heuristics=None, learning_modes=None, excluded_rolesets=None):
        Add heuristics, learning modes, or excluded rolesets to the grammar's configuration.
    set_evaluation_scores(precision: float, recall: float, f1_score: float, nr_of_correct_predictions: int, nr_of_predictions: int, nr_of_gold_standard_predictions: int):
        Set the evaluation scores for the grammar.
    add_frame_roleset_performance(frame: str, precision: float, recall: float, f1_score: float, support: int, va_frame_name: str, va_frame_id: int):
        Add the performance of a specific frame roleset.
    add_micro_evaluation_info(sen_id: int, nr_tokens: int, nr_frames: int, nr_aux: int, nr_roles: int, nr_core: int, nr_argm: int, time: int, precision: int, recall: int, f1_score: int, nr_of_correct_predictions: int, nr_of_predictions: int, nr_of_gold_standard_predictions: int):
        Add micro-evaluation information to the grammar.
    get_grammar_predicted_frames(sentence_id: Tuple[int, int]) -> Optional[List[Dict[str, any]]]:
        Retrieve the predicted frames for a specific sentence based on the grammar.
    get_micro_evaluation_info(sentence_id: Tuple[int, int]) -> Optional[Dict[str, any]]:
        Retrieve the micro-evaluation information for a specific sentence.

    The class provides functionality to manipulate and retrieve grammar information, and visualize its prediction and performance.
    """

    grammar_id: int
    grammar_size: int
    config: Dict[str, Optional[List[str]]] = field(
        default_factory=lambda: {
            "heuristics": None,
            "learning_modes": None,
            "excluded_rolesets": None,
        }
    )
    evaluation_scores: Optional[Dict[str, float]] = field(default_factory=dict)
    frame_roleset_performance: Optional[Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )
    micro_evaluation_info: Optional[List[Dict[str, any]]] = field(default_factory=list)

    def __init__(
        self, grammar_id: int, config: str, grammar_size: Optional[int] = None
    ):
        self.grammar_id = grammar_id
        self.grammar_size = grammar_size
        self.config = self._parse_config(config)
        self.evaluation_scores: Dict[str, float] = {}
        self.frame_roleset_performance: Dict[str, Dict[str, float]] = {}
        self.micro_evaluation_info: List[Dict[str, any]] = []
        self.frame_role_data: Dict[Tuple[int, int], List[Dict[str, any]]] = {}
        self.predictions: Dict[Tuple[int, int], Prediction] = {}

    def get_summary(
        self,
        sentence_id: Tuple[int, int],
        include_evaluation_scores: bool = False,
        include_micro_evaluation_info: bool = False,
    ) -> Dict[str, Any]:
        prediction = self.get_prediction(sentence_id)
        summary = {
            "grammar_id": self.grammar_id,
            "config": self.config,
        }

        if include_evaluation_scores:
            summary["evaluation_scores"] = self.evaluation_scores

        if include_micro_evaluation_info:
            if prediction is not None:
                f1_score = prediction.micro_evaluation_data.get("f1_score", 0)
            else:
                f1_score = 0
            summary["micro_evaluation_info"] = {"f1_score": f1_score.__round__(3)}

        return summary

    # Add a helper method to parse the config string

    @staticmethod
    def _parse_config(config: str) -> Dict[str, Optional[List[str]]]:
        config_dict = {
            "heuristics": None,
            "learning_modes": None,
            "excluded_rolesets": None,
        }
        for item in config.split("; "):
            key, values = item.split(": ")
            config_dict[key] = values.split(", ") if values else None
        return config_dict

    def add_prediction(self, sen_id: Tuple[int, int], predictions: Prediction):
        self.predictions[sen_id] = predictions

    def get_prediction(self, sen_id: Tuple[int, int]) -> Optional[Prediction]:
        return self.predictions.get(sen_id)

    def add_data(self, heuristics=None, learning_modes=None, excluded_rolesets=None):
        if heuristics:
            if not self.config["heuristics"]:
                self.config["heuristics"] = []
            self.config["heuristics"].extend(heuristics)

        if learning_modes:
            if not self.config["learning_modes"]:
                self.config["learning_modes"] = []
            self.config["learning_modes"].extend(learning_modes)

        if excluded_rolesets:
            if not self.config["excluded_rolesets"]:
                self.config["excluded_rolesets"] = []
            self.config["excluded_rolesets"].extend(excluded_rolesets)

    def set_evaluation_scores(
        self,
        precision: float,
        recall: float,
        f1_score: float,
        nr_of_correct_predictions: int,
        nr_of_predictions: int,
        nr_of_gold_standard_predictions: int,
    ):
        self.evaluation_scores = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "nr_of_correct_predictions": nr_of_correct_predictions,
            "nr_of_predictions": nr_of_predictions,
            "nr_of_gold_standard_predictions": nr_of_gold_standard_predictions,
        }

    def add_frame_roleset_performance(
        self,
        frame: str,
        precision: float,
        recall: float,
        f1_score: float,
        support: int,
        va_frame_name: str,
        va_frame_id: int,
    ):
        self.frame_roleset_performance[frame] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
            "va_frame_name": va_frame_name,
            "va_frame_id": va_frame_id,
        }

    def add_micro_evaluation_info(
        self,
        sen_id: int,
        nr_tokens: int,
        nr_frames: int,
        nr_aux: int,
        nr_roles: int,
        nr_core: int,
        nr_argm: int,
        time: int,
        precision: int,
        recall: int,
        f1_score: int,
        nr_of_correct_predictions: int,
        nr_of_predictions: int,
        nr_of_gold_standard_predictions: int,
    ):
        self.micro_evaluation_info.append(
            {
                "sen_id": sen_id,
                "nr_tokens": nr_tokens,
                "nr_frames": nr_frames,
                "nr_aux": nr_aux,
                "nr_roles": nr_roles,
                "nr_core": nr_core,
                "nr_argm": nr_argm,
                # "same_lemma": same_lemma,
                # "same_verb": same_verb,
                "time": time,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "nr_of_correct_predictions": nr_of_correct_predictions,
                "nr_of_predictions": nr_of_predictions,
                "nr_of_gold_standard_predictions": nr_of_gold_standard_predictions,
            }
        )

    def get_grammar_predicted_frames(
        self, sentence_id: Tuple[int, int]
    ) -> Optional[List[Dict[str, any]]]:
        if sentence_id in self.frame_role_data:
            return self.frame_role_data[sentence_id]
        return None

    def get_micro_evaluation_info(
        self, sentence_id: Tuple[int, int]
    ) -> Optional[Dict[str, any]]:
        for micro_eval_info in self.micro_evaluation_info:
            if micro_eval_info["sen_id"] == sentence_id:
                return micro_eval_info
        return None


class GrammarManager:
    """
    This GrammarManager class is used to manage the different grammars being used and evaluated.

    Attributes:
        grammar_dict (Dict[int, Grammar]): A dictionary holding grammar IDs as keys and Grammar objects as values.
        grammar_size_dict (Dict[int, int]): A dictionary holding grammar IDs as keys and the respective grammar sizes as values.
        corpus (Corpus): A Corpus object containing sentences and other linguistic data to be analyzed.

    Methods:
        __init__(self, corpus: Corpus): Initializes the GrammarManager with a specified Corpus object.
        get_grammar(self, grammar_id: int): Fetches the Grammar object from the grammar_dict by its ID.
        add_data_to_grammar(self, grammar_id: int, heuristics=None, learning_modes=None, excluded_rolesets=None):
            Adds data to a specific Grammar object.
        add_grammar(self, grammar: Grammar): Adds a new Grammar object to the grammar_dict.
        get_gold_standard_frames(self, sentence_id: Tuple[int, int]): Retrieves gold standard frames for a given sentence.
        analyze_sentence_performance(self, cv_cutoff: float = 0.7): Analyzes sentence performance using F1 scores and other metrics.
        export_sentence_summary(self, sort_by: list): Exports a summary of sentence analysis, sorted by given parameters.
        analyze_frame_performance_weighted(self, cv_cutoff: float = 0.7, frame_type: str = "va_frame_name"):
            Analyzes frame performance using weighted F1 scores and other metrics.
        export_frame_summary(self, sort_by: list): Exports a summary of frame analysis, sorted by given parameters.
        get_frame_info_for_va_frame(self, va_frame_name: str): Retrieves information about a specific frame by its name.
    """

    def __init__(self, corpus: Corpus):
        self.grammar_dict: Dict[int, Grammar] = {}
        self.grammar_size_dict: Dict[int, int] = {}
        self.corpus = corpus

    def get_grammar(self, grammar_id: int) -> Grammar:
        if grammar_id not in self.grammar_dict:
            grammar_size = self.grammar_size_dict.get(grammar_id)
            if grammar_size is not None:
                self.grammar_dict[grammar_id] = Grammar(grammar_id, grammar_size)
        return self.grammar_dict[grammar_id]

    def add_data_to_grammar(
        self,
        grammar_id: int,
        heuristics=None,
        learning_modes=None,
        excluded_rolesets=None,
    ):
        grammar = self.get_grammar(grammar_id)
        grammar.add_data(heuristics, learning_modes, excluded_rolesets)

    def add_grammar(self, grammar: Grammar):
        self.grammar_dict[grammar.grammar_id] = grammar

    def get_sorted_grammars_for_sentence(
        self, sentence_id: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        grammars_and_scores = []

        for grammar in self.grammar_dict.values():
            prediction = grammar.get_prediction(sentence_id)
            # If the grammar has no prediction for the sentence, the f1_score is 0
            if prediction is not None:
                f1_score = prediction.micro_evaluation_data.get("f1_score", 0)
                grammars_and_scores.append((grammar, f1_score))
            else:
                f1_score = 0
                grammars_and_scores.append((grammar, f1_score))

        grammars_and_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_grammars = [grammar for grammar, _ in grammars_and_scores]

        sorted_summaries = []
        for grammar in sorted_grammars:
            summary = grammar.get_summary(
                sentence_id,
                include_evaluation_scores=True,
                include_micro_evaluation_info=True,
            )
            prediction = grammar.get_prediction(sentence_id)
            if prediction is not None:
                summary["prediction"] = prediction
                sorted_summaries.append(summary)
            else:
                sorted_summaries.append(summary)

        return sorted_summaries

    def get_gold_standard_frames(
        self, sentence_id: Tuple[int, int]
    ) -> Optional[List[Dict[str, any]]]:
        if sentence_id in self.sentences:
            return self.sentences[sentence_id].frames
        return None

    def get_random_prediction(self) -> Optional[Prediction]:
        grammar = random.choice(list(self.grammar_dict.values()))
        if grammar and grammar.predictions:
            return random.choice(list(grammar.predictions.values()))
        return None

    def analyze_sentence_performance(self, cv_cutoff: float = 0.7):
        sentence_data = []

        for grammar_id, grammar in self.grammar_dict.items():
            for sentence_info in grammar.micro_evaluation_info:
                sentence_data.append(
                    {
                        "grammar_id": grammar_id,
                        "sen_id": sentence_info["sen_id"],
                        "f1_score": sentence_info["f1_score"],
                        "nr_frames": sentence_info["nr_frames"],
                    }
                )

        sentence_df = pd.DataFrame(sentence_data)

        # Calculate the mean and standard deviation for each sentence
        sentence_stats = (
            sentence_df.groupby("sen_id")
            .agg({"f1_score": ["mean", "std"], "nr_frames": "first"})
            .reset_index()
        )

        sentence_stats.columns = [
            "sen_id",
            "mean_f1_score",
            "std_f1_score",
            "nr_frames",
        ]

        # Calculate the weight factor for each sentence based on the number of frames
        sentence_stats["weight"] = 1 + np.log(sentence_stats["nr_frames"])

        # Calculate the weighted f1_score
        sentence_stats["weighted_f1_score"] = (
            sentence_stats["mean_f1_score"] * sentence_stats["weight"]
        )

        # Normalize the weighted f1_score
        max_weighted_f1_score = sentence_stats["weighted_f1_score"].max()
        min_weighted_f1_score = sentence_stats["weighted_f1_score"].min()
        sentence_stats["normalized_weighted_f1_score"] = (
            sentence_stats["weighted_f1_score"] - min_weighted_f1_score
        ) / (max_weighted_f1_score - min_weighted_f1_score)

        # Calculate the coefficient of variation
        sentence_stats["cv"] = (
            sentence_stats["std_f1_score"] / sentence_stats["mean_f1_score"]
        )

        # Get the sentences with low coefficient of variation
        consistent_sentences = sentence_stats[(sentence_stats["cv"] < cv_cutoff)]

        # Get the top 50% best performing sentences
        best_sentences = consistent_sentences.nlargest(
            n=int(len(consistent_sentences) * 0.5),
            columns=["normalized_weighted_f1_score"],
        )

        # Get the bottom 50% worst performing sentences
        worst_sentences = consistent_sentences.nsmallest(
            n=int(len(consistent_sentences) * 0.5),
            columns=["normalized_weighted_f1_score"],
        )

        # Get the sentences with high variance (above the cutoff point)
        high_variance_sentences = sentence_stats[
            (sentence_stats["cv"] >= cv_cutoff)
        ].sort_values(by=["cv"], ascending=False)

        return best_sentences, worst_sentences, high_variance_sentences

    def export_sentence_summary(
        self,
        sort_by: list = [("normalized_weighted_f1_score", False), ("nr_frames", False)],
    ) -> Tuple[str, pd.DataFrame]:
        sentence_data = []
        for grammar in self.grammar_dict.values():
            for sentence_info in grammar.micro_evaluation_info:
                sentence_data.append(
                    {
                        "sen_id": sentence_info["sen_id"],
                        "f1_score": sentence_info["f1_score"],
                        "nr_frames": sentence_info["nr_frames"],
                    }
                )

        sentence_df = pd.DataFrame(sentence_data)

        # Calculate the mean and standard deviation for each sentence
        sentence_stats = (
            sentence_df.groupby("sen_id")
            .agg({"f1_score": ["mean", "std"], "nr_frames": "first"})
            .reset_index()
        )

        sentence_stats.columns = [
            "sen_id",
            "mean_f1_score",
            "std_f1_score",
            "nr_frames",
        ]

        # Calculate the weight factor for each sentence based on the number of frames
        sentence_stats["weight"] = 1 + np.log(sentence_stats["nr_frames"])

        # Calculate the weighted f1_score
        sentence_stats["weighted_f1_score"] = (
            sentence_stats["mean_f1_score"] * sentence_stats["weight"]
        )

        # Normalize the weighted f1_score
        max_weighted_f1_score = sentence_stats["weighted_f1_score"].max()
        min_weighted_f1_score = sentence_stats["weighted_f1_score"].min()
        sentence_stats["normalized_weighted_f1_score"] = (
            sentence_stats["weighted_f1_score"] - min_weighted_f1_score
        ) / (max_weighted_f1_score - min_weighted_f1_score)

        # Calculate the coefficient of variation
        sentence_stats["cv"] = (
            sentence_stats["std_f1_score"] / sentence_stats["mean_f1_score"]
        )

        # Get the best, worst, and high variance sentences
        (
            best_sentences,
            worst_sentences,
            high_variance_sentences,
        ) = self.analyze_sentence_performance()

        # Add a 'group' column to store the group names
        sentence_stats["group"] = None

        # Assign the group names to the respective rows
        for group, group_df in [
            ("Best", best_sentences),
            ("Worst", worst_sentences),
            ("High Variance", high_variance_sentences),
        ]:
            sentence_stats.loc[
                sentence_stats["sen_id"].isin(group_df["sen_id"]), "group"
            ] = group

        # Sort the DataFrame by the provided sorting parameters
        sentence_stats = sentence_stats.sort_values(
            by=[col for col, _ in sort_by], ascending=[asc for _, asc in sort_by]
        ).reset_index(drop=True)

        output = ""
        for _, row in sentence_stats.iterrows():
            sen_id = row["sen_id"]
            mean_f1_score = row["mean_f1_score"]
            std_f1_score = row["std_f1_score"]
            nr_frames = row["nr_frames"]
            group = row["group"]
            cv = row["cv"]

            output += f"Sentence ID: {sen_id}, F1: {mean_f1_score:.2f}, std: {std_f1_score:.2f}, nr_frames: {nr_frames}, group: {group}, cv: {cv:.2f}\n"

        return output, sentence_stats

    def analyze_frame_performance_weighted(
        self, cv_cutoff: float = 0.7, frame_type: str = "va_frame_name"
    ):
        frame_data = []

        for grammar_id, grammar in self.grammar_dict.items():
            for frame, prediction_data in grammar.frame_roleset_performance.items():
                frame_data.append(
                    {
                        "grammar_id": grammar_id,
                        "frame": frame
                        if frame_type == "frame"
                        else prediction_data["va_frame_name"],
                        "f1_score": prediction_data["f1_score"],
                        "support_value": prediction_data["support"],
                    }
                )

        frame_df = pd.DataFrame(frame_data)

        # Add the weighted_f1 column
        frame_df["weighted_f1"] = frame_df["f1_score"] * frame_df["support_value"]

        # Remove rows with the frame name that contain "Average"
        frame_df = frame_df[~frame_df["frame"].str.contains("average")]

        # Remove rows with the frame name "unknown"
        frame_df = frame_df[frame_df["frame"] != "unknown"]

        # Calculate the mean and standard deviation for each frame, and sum the weighted_f1 and support_value columns
        frame_stats = (
            frame_df.groupby("frame")
            .agg(
                {
                    "f1_score": ["mean", "std"],
                    "weighted_f1": "sum",
                    "support_value": "sum",
                }
            )
            .reset_index()
        )

        frame_stats.columns = [
            "frame",
            "mean_f1_score",
            "std_f1_score",
            "sum_weighted_f1",
            "support_value",
        ]

        # Calculate the weighted mean F1 score
        frame_stats["weighted_mean_f1_score"] = (
            frame_stats["sum_weighted_f1"] / frame_stats["support_value"]
        )

        # Calculate the coefficient of variation
        frame_stats["cv"] = frame_stats["std_f1_score"] / frame_stats["mean_f1_score"]

        # Get the frames with low coefficient of variation
        consistent_frames = frame_stats[(frame_stats["cv"] < cv_cutoff)]

        # Get the top 50% best performing frames
        best_frames = consistent_frames.nlargest(
            n=int(len(consistent_frames) * 0.5),
            columns=["weighted_mean_f1_score", "support_value"],
        )

        # Get the bottom 50% worst performing frames
        worst_frames = consistent_frames.nsmallest(
            n=int(len(consistent_frames) * 0.5),
            columns=["weighted_mean_f1_score", "support_value"],
        )

        # Get the frames with high variance (above the cutoff point)
        high_variance_frames = frame_stats[
            (frame_stats["cv"] >= cv_cutoff)
        ].sort_values(by=["cv", "support_value"], ascending=False)

        return best_frames, worst_frames, high_variance_frames

    def export_frame_summary(
        self, sort_by: list = [("weighted_mean_f1_score", False), ("support", False)]
    ) -> Tuple[str, pd.DataFrame]:
        frame_data = []
        for grammar in self.grammar_dict.values():
            for frame, prediction_data in grammar.frame_roleset_performance.items():
                frame_data.append(
                    {
                        "va_frame_name": prediction_data["va_frame_name"],
                        "va_frame_id": prediction_data["va_frame_id"],
                        "f1_score": prediction_data["f1_score"],
                        "support": prediction_data["support"],
                    }
                )

        frame_df = pd.DataFrame(frame_data)
        frame_df = frame_df[~frame_df["va_frame_name"].str.contains("average")]

        # Calculate the weighted mean F1-score
        frame_df["weighted_f1"] = frame_df["f1_score"] * frame_df["support"]
        summary_df = (
            frame_df.groupby(["va_frame_name", "va_frame_id"])
            .agg({"weighted_f1": "sum", "support": "sum"})
            .reset_index()
        )
        summary_df["weighted_mean_f1_score"] = (
            summary_df["weighted_f1"] / summary_df["support"]
        )

        # Remove the weighted_f1 column
        summary_df = summary_df.drop(columns=["weighted_f1"])

        # Get the best, worst, and high variance frames
        (
            best_frames,
            worst_frames,
            high_variance_frames,
        ) = self.analyze_frame_performance_weighted()

        # Add a 'group' and 'cv' columns to store the group names and cv values
        summary_df["group"] = None
        summary_df["cv"] = None

        # Assign the group names and cv values to the respective rows
        for group, group_df in [
            ("Best", best_frames),
            ("Worst", worst_frames),
            ("High Variance", high_variance_frames),
        ]:
            summary_df.loc[
                summary_df["va_frame_name"].isin(group_df["frame"]), "group"
            ] = group
            filtered_df = summary_df.loc[
                summary_df["va_frame_name"].isin(group_df["frame"])
            ]
            summary_df.loc[filtered_df.index, "cv"] = (
                group_df.set_index("frame")["cv"]
                .loc[filtered_df["va_frame_name"]]
                .values
            )

        # Sort the DataFrame by the provided sorting parameters
        summary_df = summary_df.sort_values(
            by=[col for col, _ in sort_by], ascending=[asc for _, asc in sort_by]
        ).reset_index(drop=True)

        output = ""
        for _, row in summary_df.iterrows():
            va_frame_name = row["va_frame_name"]
            va_frame_id = row["va_frame_id"]
            mean_f1_score = row["weighted_mean_f1_score"]
            support = row["support"]
            group = row["group"]
            cv = row["cv"]

            output += f"{va_frame_name} (ID: {va_frame_id}, F1: {mean_f1_score:.2f}, support: {support}, group: {group}, cv: {f'{cv:.2f}' if cv is not None else 'NaN'})\n"

        return output, summary_df

    def get_frame_info_for_va_frame(self, va_frame_name: str) -> pd.DataFrame:
        frame_data = []
        for grammar in self.grammar_dict.values():
            for frame, prediction_data in grammar.frame_roleset_performance.items():
                frame_data.append(
                    {
                        "frame_name": frame,
                        "va_frame_name": prediction_data["va_frame_name"],
                        "va_frame_id": prediction_data["va_frame_id"],
                        "f1_score": prediction_data["f1_score"],
                        "support": prediction_data["support"],
                    }
                )

        frame_df = pd.DataFrame(frame_data)
        frame_df = frame_df[~frame_df["frame_name"].str.contains("average")]

        # Calculate the weighted mean F1-score
        frame_df["weighted_f1"] = frame_df["f1_score"] * frame_df["support"]
        summary_df = (
            frame_df.groupby(["frame_name", "va_frame_name", "va_frame_id"])
            .agg({"weighted_f1": "sum", "support": "sum"})
            .reset_index()
        )
        summary_df["mean_f1_score"] = summary_df["weighted_f1"] / summary_df["support"]

        # Filter the DataFrame to get the rows with the specified va_frame_name
        filtered_df = summary_df[summary_df["va_frame_name"] == va_frame_name]

        # Select only the required columns (frame_name, mean_f1_score, and support) and reset the index
        result_df = filtered_df[["frame_name", "mean_f1_score", "support"]].reset_index(
            drop=True
        )

        # Sort the DataFrame by the support in descending order
        result_df = result_df.sort_values(by=["support"], ascending=False).reset_index(
            drop=True
        )

        return result_df

    def get_macro_evaluation_summary(self):
        evaluation_scores_data = []

        for grammar in self.grammar_dict.values():
            evaluation_scores = grammar.evaluation_scores
            if evaluation_scores:
                evaluation_scores_data.append(evaluation_scores)

        # Create a DataFrame with the evaluation_scores_data
        df_evaluation_scores = pd.DataFrame(evaluation_scores_data)

        # Calculate the summary statistics for precision, recall, and f1_score
        summary_statistics = (
            df_evaluation_scores[["precision", "recall", "f1_score"]]
            .astype(float)
            .describe()
        )

        return summary_statistics
