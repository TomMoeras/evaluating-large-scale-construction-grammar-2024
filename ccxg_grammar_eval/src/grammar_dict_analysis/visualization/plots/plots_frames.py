import plotly.express as px
from typing import Dict
from typing import Dict, Tuple
import plotly.figure_factory as ff
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.cluster.hierarchy import linkage
import plotly.subplots as sp
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from typing import Dict, Optional, List
import sys  # nopep8
from pathlib import Path  # nopep8

# Add the 'src' directory to sys.path
src_path = Path(__file__).resolve().parent.parent.parent.parent  # nopep8
sys.path.append(str(src_path))  # nopep8
# Import the necessary module components
from grammar_corpus.corpus import Corpus, Frame, Sentence, VerbAtlasFrames
from grammar_corpus.grammar_objects import *
from grammar_dict_analysis.performance import *


def analyze_frame_performance_weighted(
    grammar_dict: Dict[int, Grammar], cv_cutoff: float = 1, frame_type: str = "frame"
):
    frame_data = []

    for grammar_id, grammar in grammar_dict.items():
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
            {"f1_score": ["mean", "std"], "weighted_f1": "sum", "support_value": "sum"}
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
    high_variance_frames = frame_stats[(frame_stats["cv"] >= cv_cutoff)].sort_values(
        by=["cv", "support_value"], ascending=False
    )

    return best_frames, worst_frames, high_variance_frames


def plot_frame_performance(grammar_dict, plot_theme: str = "plotly"):
    pio.templates.default = plot_theme
    # Get the best, worst, and high variance frames
    (
        best_frames,
        worst_frames,
        high_variance_frames,
    ) = analyze_frame_performance_weighted(
        grammar_dict, frame_type="va_frame_name", cv_cutoff=0.7
    )

    # Combine the three DataFrames into a single one
    combined_df = pd.concat([best_frames, worst_frames, high_variance_frames])

    # Assign a label to each group
    combined_df["group"] = "Best"
    combined_df.loc[worst_frames.index, "group"] = "Worst"
    combined_df.loc[high_variance_frames.index, "group"] = "High Variance"

    # Find the top 5 frames with the largest support values
    top_5_support = combined_df.nlargest(8, "support_value")

    # Create a scatter plot using Plotly with increased point sizes
    fig = px.scatter(
        combined_df,
        x="cv",
        y="weighted_mean_f1_score",  # Use the weighted mean F1 score
        size="support_value",
        size_max=30,  # Increase the maximum point size
        color="group",
        color_discrete_map={  # Define custom colors for each group
            "Best": "lime",
            "Worst": "red",
            "High Variance": "orange",
        },
        hover_name="frame",
        title="Frame Performance",
        labels={
            "weighted_mean_f1_score": "Weighted Mean F1 Score",  # Update the label
            "cv": "Coefficient of Variation (CV)",
            "support_value": "Support Value",
            "frame": "Frame",
        },
        range_x=[0, 1.5],
    )  # Show only the CV values between 0 and 2

    # Add text annotations with vertical lines for the top 5 frames with the largest support values
    for idx, (_, row) in enumerate(top_5_support.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[row["cv"], row["cv"]],
                y=[
                    row["weighted_mean_f1_score"],  # Use the weighted mean F1 score
                    row["weighted_mean_f1_score"] + 0.15 * (idx + 1),
                ],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[row["cv"]],
                y=[
                    row["weighted_mean_f1_score"] + 0.15 * (idx + 1)
                ],  # Use the weighted mean F1 score
                text=[row["frame"]],
                mode="text",
                textposition="top center",
                textfont=dict(color="black"),
                showlegend=False,
            )
        )

    # Show the plot
    return fig
