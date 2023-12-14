from typing import Dict, Optional, List
from typing import Dict, List, Union
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from multiprocessing import Pool
from tabulate import tabulate
import pickle
from time import sleep

import sys  # nopep8
from pathlib import Path  # nopep8

# Add the 'src' directory to sys.path
src_path = Path(__file__).resolve().parent.parent.parent  # nopep8
sys.path.append(str(src_path))  # nopep8

# Add the imports from the 'grammar_objects' module
from grammar_corpus import Grammar, Prediction


import tqdm
from tqdm import tqdm

# Function to get all unique sen_ids from micro_evaluation_info


def get_unique_sen_ids(grammar_dict: Dict[int, Grammar]) -> List[int]:
    sen_ids = set()
    for grammar in grammar_dict.values():
        for micro_eval_info in grammar.micro_evaluation_info:
            sen_ids.add(micro_eval_info["sen_id"])
    return list(sen_ids)


# Function to get all unique va_frame_names from frame_roleset_performance
def get_unique_va_frame_names(grammar_dict: Dict[int, Grammar]) -> List[str]:
    va_frame_names = set()
    for grammar in grammar_dict.values():
        for frame_performance in grammar.frame_roleset_performance.values():
            va_frame_names.add(frame_performance["va_frame_name"])
    return list(va_frame_names)


class ResultTable:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_result(self) -> pd.DataFrame:
        return self.df


class PerformanceParameters:
    def __init__(self, grammar_dict: Dict[int, Grammar]):
        self.grammar_dict = grammar_dict
        self.result_tables_sen_id = {}
        self.result_tables_va_frame_name = {}

        # Generate result tables for all sen_id's and va_frame_names during initialization
        self.generate_all_result_tables()

    def generate_all_result_tables(self):
        # Get all unique sen_id's and va_frame_names
        sen_id_list = get_unique_sen_ids(self.grammar_dict)
        va_frame_name_list = get_unique_va_frame_names(self.grammar_dict)

        # Iterate through sen_ids with progress bar
        for sen_id in tqdm(sen_id_list, desc="Processing sen_ids", unit="sen_id"):
            # Generate a ResultTable for each sen_id
            result_table = self.generate_result_table(sen_id=sen_id)

            # Store the ResultTable instance in the dictionary
            self.result_tables_sen_id[sen_id] = result_table

        # Iterate through va_frame_names with progress bar
        for va_frame_name in tqdm(
            va_frame_name_list, desc="Processing va_frame_names", unit="va_frame_name"
        ):
            # Generate a ResultTable for each va_frame_name
            result_table = self.generate_result_table(va_frame_name=va_frame_name)

            # Store the ResultTable instance in the dictionary
            self.result_tables_va_frame_name[va_frame_name] = result_table

    def generate_result_table(
        self, sen_id: Optional[int] = None, va_frame_name: Optional[str] = None
    ) -> ResultTable:
        if sen_id is not None:
            df = performance_parameters_optimized(self.grammar_dict, sen_id=sen_id)
        elif va_frame_name is not None:
            df = performance_parameters_optimized(
                self.grammar_dict, va_frame_name=va_frame_name
            )
        else:
            raise ValueError("Either sen_id or va_frame_name must be provided.")

        return ResultTable(df)

    def save_to_pickle(self, file_path: str) -> None:
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_pickle(cls, file_path: str) -> "PerformanceParameters":
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
        return obj


def collect_grammar_ids_with_config_items(
    grammar_dict: Dict[int, Grammar],
    config_items: Dict[str, List[str]],
    reverse: bool = False,
) -> List[int]:
    grammar_ids = []

    for grammar_id, grammar in grammar_dict.items():
        config_items_found = 0
        for config_category, items in config_items.items():
            if all(item in grammar.config[config_category] for item in items):
                config_items_found += 1
            else:
                break

        if reverse:
            if config_items_found != len(config_items):
                grammar_ids.append(grammar_id)
        else:
            if config_items_found == len(config_items):
                grammar_ids.append(grammar_id)

    return grammar_ids


# Example usage
# config_items = {
#     'heuristics': ['frequency'],
#     'learning_modes': ['argm_group']
# }
# grammar_ids = collect_grammar_ids_with_config_items(grammar_dict, config_items)


def transform_encode_data_optimized(
    grammar_dict: Dict[int, Grammar],
    metric: str = "f1_score",
    selected_grammar_ids: Optional[List[int]] = None,
    frame_name: Optional[str] = None,
    va_frame_name: Optional[str] = None,
    sen_id: Optional[int] = None,
    use_micro_evaluation_info: bool = False,
):
    """
    This function is responsible for transforming and encoding the data from a grammar dictionary for evaluation purposes. It takes several optional parameters to fine-tune the process.

    Args:
        grammar_dict (Dict[int, Grammar]): A dictionary where the keys are grammar IDs and the values are Grammar objects.
        metric (str, optional): The evaluation metric to be calculated. Default is "f1_score".
        selected_grammar_ids (List[int], optional): A list of selected grammar IDs for evaluation. If not provided, all grammars will be evaluated.
        frame_name (str, optional): If provided, the function will calculate the metric only for this specific frame name.
        va_frame_name (str, optional): If provided, the function will calculate a weighted mean of the metric for the specified VerbAtlas frame.
        sen_id (int, optional): If provided, the function will calculate the metric only for the sentence with this ID.
        use_micro_evaluation_info (bool, optional): If True, the function will use detailed evaluation information at the sentence level (micro evaluation info). Default is False.

    Returns:
        pandas.DataFrame: A DataFrame where each row corresponds to a grammar, each column corresponds to a heuristic, learning mode, or an evaluation metric, and each cell contains the value of the corresponding metric for the corresponding grammar. Cells corresponding to heuristics and learning modes are binary (1 if the heuristic/mode is used, 0 otherwise).
    """

    columns_set = set()
    data = []

    for grammar_id, grammar in grammar_dict.items():
        if selected_grammar_ids is not None and grammar_id not in selected_grammar_ids:
            continue

        heuristics = grammar.config["heuristics"]
        learning_modes = grammar.config["learning_modes"]
        excluded_rolesets = grammar.config["excluded_rolesets"]
        columns_set.update(heuristics, learning_modes)

        if metric == "diff_precision_recall":
            if frame_name or va_frame_name:
                precision = float(frame_prediction["precision"])
                recall = float(frame_prediction["recall"])
            elif sen_id is not None:
                precision = float(sentence_info["precision"])
                recall = float(sentence_info["recall"])
            else:
                precision = float(grammar.evaluation_scores["precision"])
                recall = float(grammar.evaluation_scores["recall"])
            value = abs(precision - recall)
        elif frame_name:
            frame_prediction = grammar.frame_roleset_performance.get(frame_name)
            if frame_prediction:
                value = float(frame_prediction[metric])
            else:
                continue
        elif va_frame_name:
            frame_data = []
            for frame, prediction_data in grammar.frame_roleset_performance.items():
                frame_data.append(
                    {
                        "va_frame_name": prediction_data["va_frame_name"],
                        "va_frame_id": prediction_data["va_frame_id"],
                        # Use the 'metric' variable instead of 'f1_score'
                        metric: float(prediction_data[metric]),
                        "support": prediction_data["support"],
                    }
                )

            frame_df = pd.DataFrame(frame_data)
            frame_df = frame_df[~frame_df["va_frame_name"].str.contains("average")]

            # Calculate the weighted mean metric
            frame_df["weighted_metric"] = frame_df[metric] * frame_df["support"]
            summary_df = (
                frame_df.groupby(["va_frame_name", "va_frame_id"])
                .agg({"weighted_metric": "sum", "support": "sum"})
                .reset_index()
            )
            summary_df["weighted_mean_metric"] = (
                summary_df["weighted_metric"] / summary_df["support"]
            )

            # Remove the weighted_metric column
            summary_df = summary_df.drop(columns=["weighted_metric"])

            # Rename the "weighted_mean_metric" column to the selected metric
            summary_df = summary_df.rename(columns={"weighted_mean_metric": metric})

            # Filter the DataFrame based on the given va_frame_name
            filtered_df = summary_df[summary_df["va_frame_name"] == va_frame_name]

            # Calculate the mean metric score for the given va_frame_name
            value = filtered_df[metric].mean()

            # If there are no matching frames, continue to the next iteration
            if pd.isna(value):
                continue

        elif sen_id is not None:
            sentence_info = next(
                (
                    info
                    for info in grammar.micro_evaluation_info
                    if info["sen_id"] == sen_id
                ),
                None,
            )
            if sentence_info:
                value = float(sentence_info[metric])
            else:
                continue
        elif metric == "time_out":
            sentences_over_60 = sum(
                1 for info in grammar.micro_evaluation_info if info["time"] > 60
            )
            value = sentences_over_60
        else:
            if metric != "time":
                if use_micro_evaluation_info:
                    values = [info[metric] for info in grammar.micro_evaluation_info]
                    filtered_values = [v for v in values if not np.isnan(v)]
                    value = np.mean(filtered_values)
                else:
                    value = float(grammar.evaluation_scores[metric])
            else:
                if use_micro_evaluation_info:
                    filtered_times = (
                        info["time"]
                        for info in grammar.micro_evaluation_info
                        if info["time"] <= 100
                    )
                    value = np.mean(list(filtered_times))
                else:
                    filtered_times = (
                        info["time"]
                        for info in grammar.micro_evaluation_info
                        if info["time"] <= 100
                    )
                    value = np.mean(list(filtered_times))

        new_row = {**{h: 1 for h in heuristics}, **{l: 1 for l in learning_modes}}

        if excluded_rolesets == ["nil"]:
            new_row["no_excluded_rolesets"] = 1
            columns_set.add("no_excluded_rolesets")
        elif excluded_rolesets == ["aux"]:
            new_row["aux_excluded"] = 1
            columns_set.add("aux_excluded")

        new_row[metric] = value
        columns_set.add(metric)

        data.append(new_row)

    columns_list = list(columns_set)
    transf_df = pd.DataFrame(data, columns=columns_list)
    encoded_df = pd.get_dummies(transf_df).fillna(0)

    return encoded_df


def performance_parameters_optimized(
    grammar_dict: Dict[int, Grammar],
    metric: str = "f1_score",
    func_list=["mean", "max", "min"],
    level: int = 1,
    selected_grammar_ids: Optional[List[int]] = None,
    frame_name: Optional[str] = None,
    va_frame_name: Optional[str] = None,
    sen_id: Optional[int] = None,
    full_table: bool = False,
    weights: Tuple[float, float, float] = (0.5, 0.7, 0.3),
):
    if metric not in ["precision", "recall", "f1_score", "time", "time_out"]:
        raise ValueError(
            "Invalid metric. Choose 'precision', 'recall', 'f1_score', 'time', or 'time_out'."
        )

    df = transform_encode_data_optimized(
        grammar_dict, metric, selected_grammar_ids, frame_name, va_frame_name, sen_id
    )

    always_on_columns = [
        col for col in df.columns if (col != metric) and (df[col].sum() == len(df))
    ]
    df = df.drop(always_on_columns, axis=1)

    tables = {}
    for func in func_list:
        if level == 1:
            columns = [col for col in df.columns if col != metric]
            table = pd.DataFrame(
                {
                    col: [
                        df.loc[df[col] == 1, metric].agg(func),
                        df.loc[df[col] == 0, metric].agg(func),
                        df.loc[df[col] == 1, metric].agg(func)
                        - df.loc[df[col] == 0, metric].agg(func),
                    ]
                    for col in columns
                }
            )
            table = table.T
            table.columns = ["On", "Off", "Diff"]
            table.columns = [f"{col} {func.title()}" for col in table.columns]
            tables[func.title()] = table
        else:
            raise ValueError("Invalid level. Choose either 1, 2, or 3.")

    result_table = pd.concat(tables, axis=1)

    cps = 0
    for i, func in enumerate(func_list):
        cps += weights[i] * result_table[func.title()]["Diff {}".format(func.title())]

    cps = (cps * 10).round(2)

    result_table["cps"] = cps
    result_table = result_table.sort_values(by="cps", ascending=False)

    if not full_table:
        result_table = result_table[["cps"]]

    return result_table


def performance_parameters(
    grammar_dict: Dict[int, Grammar],
    metric: str = "f1_score",
    func_list=["mean", "max", "min"],
    level: int = 1,
    selected_grammar_ids: Optional[List[int]] = None,
    frame_name: Optional[str] = None,
    va_frame_name: Optional[str] = None,
    sen_id: Optional[int] = None,
    full_table: bool = False,
    weights: Tuple[float, float, float] = (0.5, 0.7, 0.3),
    use_micro_evaluation_info: bool = False,
):
    """
    This function calculates performance metrics for a given set of grammars. It uses the `transform_encode_data_optimized` function to encode the grammar data, then performs group-by operations based on the `level` parameter and calculates the specified functions of the evaluation metric for each group. It also calculates a composite performance score (CPS) for each grammar config setting based on the results.

    Args:
        grammar_dict (Dict[int, Grammar]): A dictionary where the keys are grammar IDs and the values are Grammar objects.
        metric (str, optional): The evaluation metric to be calculated. Default is "f1_score".
        func_list (list, optional): A list of string names of aggregation functions to be used for group-by operations. Default is ["mean", "max", "min"].
        level (int, optional): The level of the group-by operation (1, 2, or 3). Default is 1.
        selected_grammar_ids (List[int], optional): A list of selected grammar IDs for evaluation. If not provided, all grammars will be evaluated.
        frame_name (str, optional): If provided, the function will calculate the metric only for this specific frame name.
        va_frame_name (str, optional): If provided, the function will calculate a weighted mean of the metric for the specified VerbAtlas frame.
        sen_id (int, optional): If provided, the function will calculate the metric only for the sentence with this ID.
        full_table (bool, optional): If True, the function returns the full table with all calculated metrics. If False, only the composite performance score is returned. Default is False.
        weights (Tuple[float, float, float], optional): A tuple of weights for calculating the composite performance score. Default is (0.5, 0.7, 0.3).
        use_micro_evaluation_info (bool, optional): If True, the function will use detailed evaluation information at the sentence level (micro evaluation info). Default is False.

    Returns:
        pandas.DataFrame: A DataFrame where each row corresponds to a grammar with a specific configuration turned on or off, and columns correspond to calculated metrics or the composite performance score. The index is made up of grammar configuration setting(s), depending on the `level` parameter.

    Raises:
        ValueError: If an invalid metric or level is provided.
    """

    if metric not in [
        "precision",
        "recall",
        "f1_score",
        "time",
        "diff_precision_recall",
        "time_out",
    ]:
        raise ValueError(
            "Invalid metric. Choose 'precision', 'recall', 'f1_score', 'diff_precision_recall', 'time', or 'time_out'."
        )

    df = transform_encode_data_optimized(
        grammar_dict,
        metric,
        selected_grammar_ids,
        frame_name,
        va_frame_name,
        sen_id,
        use_micro_evaluation_info,
    )

    # Find columns where the value is always 1
    always_on_columns = [
        col for col in df.columns if (col != metric) and (df[col].sum() == len(df))
    ]

    # Drop columns where the value is always 1
    df = df.drop(always_on_columns, axis=1)

    # Create an empty dictionary to store the tables for each function
    tables = {}

    # Loop through the list of functions and create a table for each function
    for func in func_list:
        if level == 1:
            columns = df.columns.difference([metric])

            # Create a table with the function values and On/Off differences
            table = pd.DataFrame()
            for col in columns:
                # Fix for level 1: Group by each column (except the metric) and calculate the specified function of the metric
                agg_df = df.groupby([col], as_index=False)[metric].agg(func)
                off_avg = agg_df.loc[(agg_df[col] == 0), metric].mean()
                on_avg = agg_df.loc[(agg_df[col] == 1), metric].mean()
                on_off_diff = on_avg - off_avg
                table[col] = [on_avg, off_avg, on_off_diff]

            # Transpose the table and set the index name
            table = table.T
            table.columns = ["On", "Off", "Diff"]
            table.columns = [f"{col} {func.title()}" for col in table.columns]
            tables[func.title()] = table

        elif level == 2:
            columns = df.columns.difference([metric])
            col_combinations = [
                (x, y)
                for i, x in enumerate(columns)
                for j, y in enumerate(columns)
                if i < j
            ]

            table_data = []

            for col1, col2 in col_combinations:
                agg_df = df.groupby([col1, col2])[metric].agg(func).reset_index()

                off_off_value = agg_df.loc[
                    (agg_df[col1] == 0) & (agg_df[col2] == 0), metric
                ].mean()
                on_on_value = agg_df.loc[
                    (agg_df[col1] == 1) & (agg_df[col2] == 1), metric
                ].mean()

                on_off_diff = on_on_value - off_off_value

                table_data.append(
                    [f"{col1}, {col2}", on_on_value, off_off_value, on_off_diff]
                )

            table = pd.DataFrame(
                table_data,
                columns=[
                    f"Columns {func.title()}",
                    f"On {func.title()}",
                    f"Off {func.title()}",
                    f"Diff {func.title()}",
                ],
            )
            table = table.set_index(f"Columns {func.title()}")
            tables[func.title()] = table
        elif level == 3:
            columns = df.columns.difference([metric])
            col_combinations = [
                (x, y, z)
                for i, x in enumerate(columns)
                for j, y in enumerate(columns)
                for k, z in enumerate(columns)
                if i < j < k
            ]

            table_data = []

            for col1, col2, col3 in col_combinations:
                agg_df = df.groupby([col1, col2, col3])[metric].agg(func).reset_index()

                off_off_off_value = agg_df.loc[
                    (agg_df[col1] == 0) & (agg_df[col2] == 0) & (agg_df[col3] == 0),
                    metric,
                ].mean()
                on_on_on_value = agg_df.loc[
                    (agg_df[col1] == 1) & (agg_df[col2] == 1) & (agg_df[col3] == 1),
                    metric,
                ].mean()

                on_off_diff = on_on_on_value - off_off_off_value

                table_data.append(
                    [
                        f"{col1}, {col2}, {col3}",
                        on_on_on_value,
                        off_off_off_value,
                        on_off_diff,
                    ]
                )

            table = pd.DataFrame(
                table_data,
                columns=[
                    f"Columns {func.title()}",
                    f"On {func.title()}",
                    f"Off {func.title()}",
                    f"Diff {func.title()}",
                ],
            )
            table = table.set_index(f"Columns {func.title()}")
            tables[func.title()] = table
        else:
            raise ValueError("Invalid level. Choose either 1, 2 or 3.")

    result_table = pd.concat(tables, axis=1)

    # round the values to 2 decimals
    result_table = result_table.round(2)

    # Get the ordered labels from the dendrogram
    # ordered_labels = cluster_performance_parameters(result_table)

    # Reorder the result_table based on the ordered_labels
    # result_table = result_table.loc[ordered_labels]

    # Calculate the composite score using the weights
    diff_values = {}
    for i, func in enumerate(func_list):
        diff_values[func] = result_table[func.title()]["Diff {}".format(func.title())]

    cps = 0
    for i, func in enumerate(func_list):
        cps += weights[i] * diff_values[func]

    cps = cps * 10

    result_table["cps"] = cps.__round__(2)

    # Sort the table on the composite score column
    result_table = result_table.sort_values(by="cps", ascending=False)

    if not full_table:
        # Only show the composite score column
        result_table = result_table[["cps"]]

    return result_table


def get_normalized_weighted_metric(sen_id, metric, grammar_dict):
    sentence_data = []
    for grammar in grammar_dict.values():
        for sentence_info in grammar.micro_evaluation_info:
            if sentence_info["sen_id"] == sen_id:
                sentence_data.append(
                    {
                        "sen_id": sentence_info["sen_id"],
                        metric: sentence_info[metric],
                        "nr_frames": sentence_info["nr_frames"],
                    }
                )

    if not sentence_data:
        print(f"sen_id {sen_id} not found in grammar_dict")
        return None

    sentence_df = pd.DataFrame(sentence_data)
    sentence_stats = (
        sentence_df.groupby("sen_id")
        .agg({metric: ["mean", "std"], "nr_frames": "first"})
        .reset_index()
    )

    sentence_stats.columns = ["sen_id", f"mean_{metric}", f"std_{metric}", "nr_frames"]
    sentence_stats["weight"] = 1 + np.log(sentence_stats["nr_frames"])
    sentence_stats[f"weighted_{metric}"] = (
        sentence_stats[f"mean_{metric}"] * sentence_stats["weight"]
    )

    max_weighted_metric = sentence_stats[f"weighted_{metric}"].max()
    min_weighted_metric = sentence_stats[f"weighted_{metric}"].min()

    if max_weighted_metric == min_weighted_metric:
        sentence_stats[f"normalized_weighted_{metric}"] = sentence_stats[
            f"weighted_{metric}"
        ]
    else:
        sentence_stats[f"normalized_weighted_{metric}"] = (
            sentence_stats[f"weighted_{metric}"] - min_weighted_metric
        ) / (max_weighted_metric - min_weighted_metric)

    return sentence_stats[f"normalized_weighted_{metric}"].values[0]


def cluster_performance_parameters(result_table: pd.DataFrame):
    # Remove rows with missing or infinite values
    cleaned_result_table = (
        result_table.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    )

    # Calculate the condensed distance matrix
    condensed_matrix = pdist(cleaned_result_table)

    # Ensure there are no infinite or missing values in the condensed distance matrix
    assert (
        not np.isnan(condensed_matrix).any() and not np.isinf(condensed_matrix).any()
    ), "The condensed distance matrix must contain only finite values."

    # Calculate the linkage matrix
    Z = linkage(condensed_matrix, method="ward")

    # Create a dendrogram
    dendro = dendrogram(
        Z, labels=cleaned_result_table.index, orientation="left", no_plot=True
    )

    # Extract the ordered labels from the dendrogram
    ordered_labels = dendro["ivl"]

    # Return the ordered labels
    return ordered_labels
