import pickle
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from typing import Dict, Optional, Tuple, List
import sys  # nopep8
from pathlib import Path  # nopep8

# Add the 'src' directory to sys.path
src_path = Path(__file__).resolve().parent.parent.parent  # nopep8
sys.path.append(str(src_path))  # nopep8
from grammar_dict_analysis.performance import (
    PerformanceParameters,
    performance_parameters,
    ResultTable,
)
from grammar_corpus import Corpus, Grammar, GrammarManager
from grammar_dict_analysis.load_grammar_dict_pickle import (
    load_grammar_dict_from_pickle,
    GrammarDictUnpickler,
)

import pandas as pd
from typing import Dict, Optional, List


class vaFrameInfo:
    def __init__(
        self,
        name: str,
        frame_id: str,
        support: float,
        f1_score: float,
        performance_dict: dict,
    ):
        self.name = name
        self.frame_id = frame_id
        self.support = support
        self.f1_score = f1_score
        self.performance = performance_dict

    @classmethod
    def from_va_summary_and_pivoted_df(
        cls, va_summary_df: pd.DataFrame, va_pivoted_df: pd.DataFrame, frame_name: str
    ) -> "vaFrameInfo":
        summary_row = va_summary_df.loc[
            va_summary_df["va_frame_name"] == frame_name
        ].iloc[0]
        pivoted_row = va_pivoted_df.loc[frame_name]

        performance_dict = {}
        for key, value in pivoted_row.to_dict().items():
            new_key = (frame_name, key[1])
            performance_dict[new_key] = value

        return cls(
            name=frame_name,
            frame_id=summary_row["va_frame_id"],
            support=summary_row["support"],
            f1_score=summary_row["weighted_mean_f1_score"],
            performance_dict=performance_dict,
        )


class SentenceInfo:
    def __init__(
        self,
        sen_id: Tuple[int, int],
        mean_f1_score: float,
        std_f1_score: float,
        nr_frames: int,
        performance_dict: dict,
    ):
        self.sen_id = sen_id
        self.mean_f1_score = mean_f1_score
        self.std_f1_score = std_f1_score
        self.nr_frames = nr_frames
        self.performance = performance_dict

    @classmethod
    def from_sentence_summary_and_pivoted_df(
        cls,
        sentence_summary_df: pd.DataFrame,
        sen_id_pivoted_df: pd.DataFrame,
        sen_id: Tuple[int, int],
    ) -> "SentenceInfo":
        summary_row = sentence_summary_df.loc[
            sentence_summary_df["sen_id"] == sen_id
        ].iloc[0]

        pivoted_row = sen_id_pivoted_df.loc[[sen_id]]

        performance_dict = {}
        for key, value in pivoted_row.to_dict().items():
            new_key = (sen_id, key[1])
            performance_dict[new_key] = value

        return cls(
            sen_id=sen_id,
            mean_f1_score=summary_row["mean_f1_score"],
            std_f1_score=summary_row["std_f1_score"],
            nr_frames=summary_row["nr_frames"],
            performance_dict=performance_dict,
        )


class SentenceFrameInfoManager:
    def __init__(
        self,
        va_summary_df: pd.DataFrame,
        va_pivoted_df: pd.DataFrame,
        sentence_summary_df: pd.DataFrame,
        sen_id_pivoted_df: pd.DataFrame,
    ):
        self.va_frame_info_dict = self.create_va_frame_info_objects(
            va_summary_df, va_pivoted_df
        )
        self.sentence_info_dict = self.create_sentence_info_objects(
            sentence_summary_df, sen_id_pivoted_df
        )

    def create_va_frame_info_objects(
        self, va_summary_df: pd.DataFrame, va_pivoted_df: pd.DataFrame
    ) -> Dict[str, vaFrameInfo]:
        va_frame_info_dict = {}

        for frame_name in va_summary_df["va_frame_name"]:
            va_frame_info = vaFrameInfo.from_va_summary_and_pivoted_df(
                va_summary_df, va_pivoted_df, frame_name
            )
            va_frame_info_dict[frame_name] = va_frame_info

        return va_frame_info_dict

    def create_sentence_info_objects(
        self, sentence_summary_df: pd.DataFrame, sen_id_pivoted_df: pd.DataFrame
    ) -> Dict[str, vaFrameInfo]:
        sentence_info_dict = {}

        for sen_id in sentence_summary_df["sen_id"]:
            sentence_info = SentenceInfo.from_sentence_summary_and_pivoted_df(
                sentence_summary_df, sen_id_pivoted_df, sen_id
            )
            sentence_info_dict[sen_id] = sentence_info

        return sentence_info_dict

    @classmethod
    def from_grammar_dict(
        cls, grammar_dict: Dict[str, Grammar]
    ) -> "vaFrameInfoManager":
        performance_parameters = PerformanceParameters(grammar_dict)
        df = preprocess_data(performance_parameters)
        sen_id_pivoted_df, va_pivoted_df = create_parameter_vectors(df)
        manager = GrammarManager(grammar_dict)
        output, va_summary_df = manager.export_frame_summary(
            sort_by=[("support", False), ("weighted_mean_f1_score", False)]
        )
        output, sentence_summary_df = manager.export_sentence_summary(
            sort_by=[("normalized_weighted_f1_score", False), ("nr_frames", False)]
        )
        return cls(va_summary_df, va_pivoted_df, sentence_summary_df, sen_id_pivoted_df)

    @classmethod
    def from_performance_object(
        cls, performance_object: PerformanceParameters, manager: GrammarManager
    ) -> "vaFrameInfoManager":
        df = preprocess_data(performance_object)
        sen_id_pivoted_df, va_pivoted_df = create_parameter_vectors(df)
        output, va_summary_df = manager.export_frame_summary(
            sort_by=[("support", False), ("weighted_mean_f1_score", False)]
        )
        output, sentence_summary_df = manager.export_sentence_summary(
            sort_by=[("normalized_weighted_f1_score", False), ("nr_frames", False)]
        )
        return cls(va_summary_df, va_pivoted_df, sentence_summary_df, sen_id_pivoted_df)

    def save_to_pickle(self, filepath: str) -> None:
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_pickle(cls, filepath: str) -> "vaFrameInfoManager":
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def get_va_frame_info(self, frame_name: str) -> Optional[vaFrameInfo]:
        return self.va_frame_info_dict.get(frame_name, None)

    def get_sentence_info(self, sen_id: Tuple[int, int]) -> Optional[SentenceInfo]:
        return self.sentence_info_dict.get(sen_id, None)

    def get_high_setting_frames(self, setting_name: str) -> pd.DataFrame:
        high_setting_frames_data = []

        for frame_name, va_frame_info in self.va_frame_info_dict.items():
            performance_dict = va_frame_info.performance
            setting_value = performance_dict.get((frame_name, setting_name), 0)

            # Check if the setting value has the highest score and is higher than 1
            if setting_value > 1 and all(
                setting_value >= v
                for k, v in performance_dict.items()
                if k != (frame_name, setting_name)
            ):
                high_setting_frames_data.append(
                    {
                        "frame_name": va_frame_info.name,
                        # 'frame_id': va_frame_info.frame_id,
                        "support": va_frame_info.support,
                        "general_f1": va_frame_info.f1_score,
                        f"{setting_name} cps": setting_value,
                    }
                )

        high_setting_frames_df = pd.DataFrame(high_setting_frames_data)
        return high_setting_frames_df

    def get_high_setting_sentences(
        self, setting_name: str, corpus: Corpus, va_frame_name: Optional[str] = None
    ) -> pd.DataFrame:
        high_setting_sentences_data = []

        for sen_id, sentence_info in self.sentence_info_dict.items():
            # If va_frame_name is provided, check if it's present in the sentence's frames
            if va_frame_name is not None:
                sentence = corpus.sentences.get(sen_id)
                if sentence is None or not any(
                    frame.va_frame_name == va_frame_name for frame in sentence.frames
                ):
                    continue

            performance_dict = sentence_info.performance
            setting_value = performance_dict.get((sen_id, setting_name), 0)

            # Check if the setting value has the highest score and is higher than 1
            if setting_value.get(sen_id, 0) > 1 and all(
                setting_value.get(sen_id, 0) >= v
                for k, inner_dict in performance_dict.items()
                if k != (sen_id, setting_name)
                for v in inner_dict.values()
            ):
                high_setting_sentences_data.append(
                    {
                        "sen_id": sentence_info.sen_id,
                        "nr_frames": sentence_info.nr_frames,
                        "general_f1": sentence_info.mean_f1_score,
                        f"{setting_name} cps": setting_value,
                    }
                )

        high_setting_sentences_df = pd.DataFrame(high_setting_sentences_data)
        return high_setting_sentences_df

    def get_all_settings_summary(self, setting_names: List[str]) -> pd.DataFrame:
        all_settings_summary_data = []

        for setting_name in setting_names:
            high_setting_frames_df = self.get_high_setting_frames(setting_name)
            total_frames = len(high_setting_frames_df)
            total_support = high_setting_frames_df["support"].mean()

            all_settings_summary_data.append(
                {
                    "heuristic": setting_name,
                    "nr_frames": total_frames,
                    "mean_support": total_support,
                }
            )

        all_settings_summary_df = pd.DataFrame(all_settings_summary_data)

        # sort on nr_frames and mean_support
        all_settings_summary_df = all_settings_summary_df.sort_values(
            by=["nr_frames", "mean_support"], ascending=False
        )

        return all_settings_summary_df


def create_parameter_vectors(df: pd.DataFrame) -> pd.DataFrame:
    # sort the DataFrame by type
    sen_id_df = df[df["type"] == "sen_id"]
    va_df = df[df["type"] == "va_frame_name"]
    # Pivot the DataFrame to create a vector for each parameter
    sen_id_pivoted_df = sen_id_df.pivot_table(
        index=["type", "setting"], columns="id", values="cps"
    ).fillna(0)
    va_pivoted_df = va_df.pivot_table(
        index=["type", "setting"], columns="id", values="cps"
    ).fillna(0)

    return sen_id_pivoted_df.T, va_pivoted_df.T


def preprocess_data(performance_parameters: PerformanceParameters) -> pd.DataFrame:
    data = []

    # Collect data for sen_ids
    for (
        sen_id,
        result_table_sen_id,
    ) in performance_parameters.result_tables_sen_id.items():
        try:
            composite_scores = result_table_sen_id.get_result()["cps"].to_dict()
        except KeyError:
            composite_scores = {}

        for parameter, score in composite_scores.items():
            data.append(
                {"id": sen_id, "type": "sen_id", "setting": parameter, "cps": score}
            )

    # Collect data for va_frame_names
    for (
        va_frame_name,
        result_table_va_frame_name,
    ) in performance_parameters.result_tables_va_frame_name.items():
        try:
            composite_scores = result_table_va_frame_name.get_result()["cps"].to_dict()
        except KeyError:
            composite_scores = {}

        for parameter, score in composite_scores.items():
            data.append(
                {
                    "id": va_frame_name,
                    "type": "va_frame_name",
                    "setting": parameter,
                    "cps": score,
                }
            )

    # Create a DataFrame using the collected data
    df = pd.DataFrame(data)
    return df.fillna(0)  # Replace NaN values with 0


def find_sentence_info(grammar_dict: Dict[int, Grammar], sen_id: int):
    sentence_info_list = []

    for grammar_id, grammar in grammar_dict.items():
        for info in grammar.micro_evaluation_info:
            if info["sen_id"] == sen_id:
                sentence_info_list.append(info)

    return sentence_info_list if sentence_info_list else None
