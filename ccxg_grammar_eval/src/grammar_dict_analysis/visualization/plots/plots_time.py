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
from grammar_corpus.grammar_objects import Grammar, GrammarManager, Prediction
from grammar_dict_analysis.performance import *


# plot_time_analysis(grammar_dict, plot_types=['scatter', 'hexbin', 'detailed'])
def plot_time_analysis(
    grammar_dict: Dict[int, Grammar],
    selected_grammar_ids: Optional[List[int]] = None,
    plot_types: List[str] = ["scatter"],
    plot_theme: str = "plotly",
):
    # Get the micro_evaluation_info data for the selected grammars
    data = []
    for grammar_id, grammar in grammar_dict.items():
        grammar_data = grammar.micro_evaluation_info
        grammar_df = pd.DataFrame(grammar_data)

        # If the grammar contains NaN value for time, skip it
        if grammar_df["time"].isna().any():
            continue

        if selected_grammar_ids is None or grammar_id in selected_grammar_ids:
            data.extend(grammar_data)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # remove any value higher than 60 seconds
    df = df[df["time"] < 60.5]

    # remove rows that contain time value nan
    df = df[~df["time"].isna()]

    # Calculate statistics
    statistics = df["time"].describe()

    # Set default plotly theme
    pio.templates.default = plot_theme

    for plot_type in plot_types:
        if plot_type == "scatter":
            # Create scatter plot
            fig_scatter = go.Figure()

            fig_scatter.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["time"],
                    mode="markers",
                    name="Time Taken",
                    marker=dict(size=2),
                )
            )

            # Customize layout
            fig_scatter.update_layout(
                title="Time to Comprehend Analysis (Scatter)",
                xaxis_title="Sentence",
                yaxis_title="Time Taken",
                showlegend=False,
            )

            return fig_scatter

        elif plot_type == "hexbin":
            # Create hexbin plot
            fig_hexbin = go.Figure()

            fig_hexbin.add_trace(
                go.Histogram2d(
                    x=df.index,
                    y=df["time"],
                    colorscale="pubu",
                    nbinsx=100,
                    nbinsy=50,
                )
            )

            # Customize layout
            fig_hexbin.update_layout(
                title="Time to Comprehend Analysis (Hexbin)",
                xaxis_title="Sentence",
                yaxis_title="Time Taken",
                showlegend=False,
            )

            return fig_hexbin

        elif plot_type == "detailed":
            # Create subplots
            fig_detailed = sp.make_subplots(
                rows=1, cols=3, subplot_titles=["Histogram", "Box Plot", "CDF"]
            )

            # Histogram
            fig_detailed.add_trace(
                go.Histogram(x=df["time"], nbinsx=20, name="Histogram"), row=1, col=1
            )

            # Box plot
            fig_detailed.add_trace(
                go.Box(y=df["time"], name="Box Plot", boxmean=True), row=1, col=2
            )

            # CDF
            fig_detailed.add_trace(
                go.Histogram(
                    x=df["time"],
                    nbinsx=20,
                    cumulative_enabled=True,
                    histnorm="probability",
                    name="CDF",
                ),
                row=1,
                col=3,
            )

            # Customize layout
            fig_detailed.update_layout(
                title="Time to Comprehend Analysis (Detailed)",
                showlegend=False,
                width=1000,
                height=400,
            )

            # Update x and y axis labels
            fig_detailed.update_xaxes(title_text="Time (seconds)", row=1, col=1)
            fig_detailed.update_yaxes(title_text="Frequency", row=1, col=1)
            fig_detailed.update_xaxes(title_text="Time (seconds)", row=1, col=2)
            fig_detailed.update_yaxes(title_text="Time (seconds)", row=1, col=2)
            fig_detailed.update_xaxes(title_text="Time (seconds)", row=1, col=3)
            fig_detailed.update_yaxes(title_text="Cumulative Probability", row=1, col=3)

            return fig_detailed


def plot_general_time_analysis(
    grammar_dict: Dict[int, Grammar], plot_theme: str = "plotly"
):
    time_stats = time_analysis(grammar_dict)

    # Set default plotly theme
    pio.templates.default = plot_theme

    fig = go.Figure()

    for _, row in time_stats.iterrows():
        grammar = grammar_dict[row["grammar_id"]]
        heuristics = ", ".join(grammar.config["heuristics"])
        learning_modes = ", ".join(grammar.config["learning_modes"])
        excluded_rolesets = ", ".join(grammar.config["excluded_rolesets"])

        hover_template = (
            f"Grammar {row['grammar_id']}<br>"
            f"Mean Time: {row['mean_time']:.3f} ms<br>"
            f"Precision: {float(grammar.evaluation_scores['precision']):.3f}<br>"
            f"Recall: {float(grammar.evaluation_scores['recall']):.3f}<br>"
            f"F1-score: {float(grammar.evaluation_scores['f1_score']):.3f}<br>"
            f"<br>Heuristics: {heuristics}<br>"
            f"Learning Modes: {learning_modes}<br>"
            f"Excluded Rolesets: {excluded_rolesets}"
        )

        fig.add_trace(
            go.Bar(
                y=[row["grammar_id"]],
                x=[row["mean_time"]],
                text=[f"{row['mean_time']:.3f} s"],
                textposition="auto",
                hovertemplate=hover_template,
                name=f"Grammar {int(row['grammar_id'])}",
                showlegend=True,
                orientation="h",
            )
        )

    fig.update_layout(
        title="Mean Processing Time of Each Grammar",
        yaxis_title="Grammar ID",
        xaxis_title="Mean Processing Time (s)",
        yaxis=dict(type="category"),
        xaxis=dict(type="linear"),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=12),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.3)",
            title=dict(text="Configurations", font=dict(color="rgba(0, 0, 0, 0.7)")),
            font=dict(color="rgba(0, 0, 0, 0.7)"),
        ),
    )

    return fig


def plot_sentence_time(
    grammar_dict: Dict[int, Grammar],
    sen_id: Tuple[int, int],
    display_mode: str = "grouped_grammars",
    orientation: str = "vertical",
    plot_theme: str = "plotly",
):
    # Get the sentence prediction data for the specified sentence id
    sentence_data = []
    for grammar_id, grammar in grammar_dict.items():
        for info in grammar.micro_evaluation_info:
            if info["sen_id"] == sen_id:
                sentence_data.append(
                    {
                        "grammar_id": grammar_id,
                        "time": float(info["time"]),
                        "config": grammar.config,
                        "heuristics": grammar.config["heuristics"],
                        "learning_modes": grammar.config["learning_modes"],
                        "excluded_rolesets": grammar.config["excluded_rolesets"],
                    }
                )

    # Create a DataFrame
    df = pd.DataFrame(sentence_data)

    # Sort DataFrame by time in ascending order
    df = df.sort_values("grammar_id", ascending=False)

    # Set default plotly theme
    pio.templates.default = plot_theme

    # Create a bar chart
    fig = go.Figure()

    if display_mode == "grouped_grammars":
        for index, row in df.iterrows():
            hover_template = (
                f"Grammar {row['grammar_id']}<br>"
                f"Time: {row['time']:.3f}s<br>"
                f"<br>Heuristics: {row['heuristics']}<br>"
                f"Learning Modes: {row['learning_modes']}<br>"
                f"Excluded Rolesets: {row['excluded_rolesets']}"
            )

            if orientation == "vertical":
                fig.add_trace(
                    go.Bar(
                        x=[f"Grammar {row['grammar_id']}"],
                        y=[row["time"]],
                        name=f"Grammar {row['grammar_id']}",
                        text=[f"{row['time']:.3f}"],
                        textposition="auto",
                        hovertemplate=hover_template,
                    )
                )
            elif orientation == "horizontal":
                fig.add_trace(
                    go.Bar(
                        x=[row["time"]],
                        y=[f"Grammar {row['grammar_id']}"],
                        name=f"Grammar {row['grammar_id']}",
                        text=[f"{row['time']:.3f}"],
                        textposition="auto",
                        hovertemplate=hover_template,
                        orientation="h",
                    )
                )
            else:
                raise ValueError(
                    "Invalid orientation. Choose either 'vertical' or 'horizontal'."
                )

    fig.update_layout(
        title=f"Sentence {sen_id} Processing Time for Grammars",
        xaxis_title="Grammar ID" if orientation == "vertical" else "Time (s)",
        yaxis_title="Time (s)" if orientation == "vertical" else "Grammar ID",
        yaxis=dict(type="category" if orientation == "horizontal" else "linear"),
        xaxis=dict(type="category" if orientation == "vertical" else "linear"),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(size=12),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.3)",
            title=dict(text="Configurations", font=dict(color="rgba(0, 0, 0, 0.7)")),
            font=dict(color="rgba(0, 0, 0, 0.7)"),
        ),
    )

    return fig


import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict


def plot_relationships(grammar_dict: Dict[int, Grammar]) -> None:
    # Extract data from grammar_dict
    data = []
    for grammar in grammar_dict.values():
        for sentence_info in grammar.micro_evaluation_info:
            data.append(
                {
                    "nr_tokens": sentence_info["nr_tokens"],
                    "nr_frames": sentence_info["nr_frames"],
                    "nr_aux": sentence_info["nr_aux"],
                    "nr_roles": sentence_info["nr_roles"],
                    "nr_core": sentence_info["nr_core"],
                    "nr_argm": sentence_info["nr_argm"],
                    "time_out": int(sentence_info["time"] > 60),
                }
            )

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Group the data by unique variables and calculate mean values
    df_grouped = df.groupby(["nr_tokens", "nr_frames", "nr_roles"]).mean().reset_index()

    # Calculate probability of timeouts for each unique number of frames
    frame_timeouts = Counter(df[df["time_out"] == 1]["nr_frames"])
    frame_counts = Counter(df["nr_frames"])
    frame_probs = {k: frame_timeouts[k] / frame_counts[k] for k in frame_counts.keys()}
    frame_probs_df = pd.DataFrame(
        list(frame_probs.items()), columns=["nr_frames", "probability"]
    )

    # Create subplots
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "nr_frames vs nr_roles",
            "nr_frames vs probability of time-outs",
        ],
    )

    # Add scatter plots to subplots
    fig.add_trace(
        go.Scatter(
            x=df_grouped["nr_frames"],
            y=df_grouped["nr_roles"],
            mode="markers",
            name="Frames vs Roles",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=frame_probs_df["nr_frames"],
            y=frame_probs_df["probability"],
            mode="markers",
            name="Frames vs Timeouts",
        ),
        row=1,
        col=2,
    )

    # Customize layout
    fig.update_layout(title="Relationships between variables", height=400, width=800)

    # Update x and y axis labels
    fig.update_xaxes(title_text="Number of Frames", row=1, col=1)
    fig.update_yaxes(title_text="Number of Roles", row=1, col=1)
    fig.update_xaxes(title_text="Number of Frames", row=1, col=2)
    fig.update_yaxes(title_text="Probability of Timeouts", row=1, col=2)

    return fig
