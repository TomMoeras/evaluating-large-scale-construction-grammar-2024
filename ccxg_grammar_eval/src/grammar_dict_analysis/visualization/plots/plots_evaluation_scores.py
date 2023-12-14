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


def plot_macro_evaluation_scores(
    grammar_dict: Dict[int, Grammar],
    selected_grammar_ids: Optional[List[int]] = None,
    display_mode: str = "grouped_metrics",
    plot_theme: str = "plotly",
):
    # Create a DataFrame containing the evaluation scores for each grammar
    data = []

    for grammar_id, grammar in grammar_dict.items():
        if selected_grammar_ids is None or grammar_id in selected_grammar_ids:
            data.append(
                {
                    "grammar_id": grammar_id,
                    "filename": f"Grammar {grammar_id}",
                    "precision": float(grammar.evaluation_scores["precision"]),
                    "recall": float(grammar.evaluation_scores["recall"]),
                    "f1-score": float(grammar.evaluation_scores["f1_score"]),
                    "config": grammar.config,
                    "heuristics": grammar.config["heuristics"],
                    "learning_modes": grammar.config["learning_modes"],
                    "excluded_rolesets": grammar.config["excluded_rolesets"],
                }
            )

    df = pd.DataFrame(data)

    # Sort DataFrame on descending F1 score
    df = df.sort_values("f1-score", ascending=False)

    # Set default plotly theme
    pio.templates.default = plot_theme

    fig = go.Figure()

    if display_mode == "grouped_metrics":
        for index, row in df.iterrows():
            hover_template = (
                f"Grammar {row['grammar_id']}<br>"
                f"Precision: {row['precision']:.3f}<br>"
                f"Recall: {row['recall']:.3f}<br>"
                f"F1-score: {row['f1-score']:.3f}<br>"
                f"<br>Heuristics: {row['heuristics']}<br>"
                f"Learning Modes: {row['learning_modes']}<br>"
                f"Excluded Rolesets: {row['excluded_rolesets']}"
            )

            fig.add_trace(
            go.Bar(
                x=["Precision", "Recall", "F1-score"],
                y=[row["precision"], row["recall"], row["f1-score"]],
                name=row["filename"],
                hovertemplate=hover_template,
            )
        )

    elif display_mode == "grouped_grammars":
        metrics = ["Precision", "Recall", "F1-score"]
        colors = ["#0A84FF", "#FFD60A", "#FF375F"]

        for i, grammar in enumerate(df["filename"].unique()):
            grammar_data = df[df["filename"] == grammar]

            for index, (metric, color) in enumerate(zip(metrics, colors)):
                row = grammar_data.iloc[0]
                hover_template = (
                    f"Grammar {row['grammar_id']}<br>"
                    f"Precision: {row['precision']:.3f}<br>"
                    f"Recall: {row['recall']:.3f}<br>"
                    f"F1-score: {row['f1-score']:.3f}<br>"
                    f"<br>Heuristics: {row['heuristics']}<br>"
                    f"Learning Modes: {row['learning_modes']}<br>"
                    f"Excluded Rolesets: {row['excluded_rolesets']}"
                )
                fig.add_trace(
                    go.Bar(
                        x=[f"{grammar}"],
                        y=[row[f"{metric.lower()}"]],
                        name=metric,
                        marker_color=color,
                        text=[f"{row[f'{metric.lower()}']:.3f}"],
                        textposition="auto",
                        offsetgroup=index,
                        width=0.2,
                        hovertemplate=hover_template,
                        showlegend=True if i == 0 else False,
                    )
                )

    else:
        raise ValueError(
            "Invalid display_mode. Choose either 'grouped_metrics' or 'grouped_grammars'."
        )

    fig.update_layout(
        title="Precision, Recall, and F1-score for each grammar",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.1,
        showlegend=False,
    )

    return fig


def cluster_frames(grammar_dict: Dict[int, Grammar]):
    # Prepare the data
    data = []
    for grammar_id, grammar in grammar_dict.items():
        for frame_name, frame_prediction in grammar.frame_predictions.items():
            data.append(
                {
                    "Grammar ID": grammar_id,
                    "Frame Name": frame_name,
                    "Precision": float(frame_prediction["precision"]),
                    "Recall": float(frame_prediction["recall"]),
                    "F1 Score": float(frame_prediction["f1_score"]),
                    "Support": float(frame_prediction["support"]),
                }
            )

    df = pd.DataFrame(data)
    # remove rows with the frame name that contain "Average"
    df = df[~df["Frame Name"].str.contains("average")]

    # Calculate the average scores for each frame
    avg_df = df.groupby("Frame Name").agg(
        {"Precision": "mean", "Recall": "mean", "F1 Score": "mean", "Support": "mean"}
    )

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    imputed_data = imputer.fit_transform(avg_df)

    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    # Create a dendrogram with Plotly
    fig = ff.create_dendrogram(
        scaled_data,
        orientation="left",
        labels=avg_df.index,
        linkagefun=lambda x: linkage(x, method="ward", metric="euclidean"),
    )

    fig.update_layout(
        width=800,
        height=1000,
        title="Frame Clustering Dendrogram",
        xaxis_title="Distance",
        yaxis_title="Frame Name",
    )
    fig.show()

    return fig["layout"]["yaxis"]["ticktext"]


def plot_frame_f1_scores(
    grammar_dict: Dict[int, Grammar],
    frame_name: str,
    display_mode: str = "grouped_grammars",
    orientation: str = "vertical",
    plot_theme: str = "plotly_dark",
):
    # Get the frame prediction data for the specified frame
    frame_data = []
    for grammar_id, grammar in grammar_dict.items():
        frame_prediction = grammar.frame_predictions.get(frame_name)
        if frame_prediction:
            frame_data.append(
                {
                    "grammar_id": grammar_id,
                    "f1_score": float(frame_prediction["f1_score"]),
                    "precision": float(frame_prediction["precision"]),
                    "recall": float(frame_prediction["recall"]),
                    "config": grammar.config,
                    "heuristics": grammar.config["heuristics"],
                    "learning_modes": grammar.config["learning_modes"],
                    "excluded_rolesets": grammar.config["excluded_rolesets"],
                }
            )

    # Create a DataFrame
    df = pd.DataFrame(frame_data)

    # Create a bar chart
    fig = go.Figure()

    # Set default plotly theme
    pio.templates.default = plot_theme

    if display_mode == "grouped_grammars":
        for index, row in df.iterrows():
            hover_template = (
                f"Grammar {row['grammar_id']}<br>"
                f"Precision: {row['precision']:.3f}<br>"
                f"Recall: {row['recall']:.3f}<br>"
                f"F1-score: {row['f1_score']:.3f}<br>"
                f"<br>Heuristics: {row['heuristics']}<br>"
                f"Learning Modes: {row['learning_modes']}<br>"
                f"Excluded Rolesets: {row['excluded_rolesets']}"
            )

            if orientation == "vertical":
                fig.add_trace(
                    go.Bar(
                        x=[f"Grammar {row['grammar_id']}"],
                        y=[row["f1_score"]],
                        name=f"Grammar {row['grammar_id']}",
                        text=[f"{row['f1_score']:.3f}"],
                        textposition="auto",
                        hovertemplate=hover_template,
                    )
                )
            elif orientation == "horizontal":
                fig.add_trace(
                    go.Bar(
                        x=[row["f1_score"]],
                        y=[f"Grammar {row['grammar_id']}"],
                        name=f"Grammar {row['grammar_id']}",
                        text=[f"{row['f1_score']:.3f}"],
                        textposition="auto",
                        hovertemplate=hover_template,
                        orientation="h",
                    )
                )
            else:
                raise ValueError(
                    "Invalid orientation. Choose either 'vertical' or 'horizontal'."
                )

    else:
        raise ValueError("Invalid display_mode. Choose 'grouped_grammars'.")

    # Customize layout
    if orientation == "vertical":
        fig.update_layout(
            title=f"F1-Scores for Frame: {frame_name}",
            xaxis_title="Grammar ID",
            yaxis_title="F1-Score",
            showlegend=False,
        )
    elif orientation == "horizontal":
        fig.update_layout(
            title=f"F1-Scores for Frame: {frame_name}",
            xaxis_title="F1-Score",
            yaxis_title="Grammar ID",
            showlegend=False,
        )

    return fig


def plot_cumulative_average_f1_scores(
    grammar_dict: Dict[int, Grammar], batch_size: int = 1, plot_theme: str = "plotly"
):
    # Set default plotly theme
    pio.templates.default = plot_theme

    # Transform the grammar_dict into a DataFrame
    data = []
    for grammar_id, grammar in grammar_dict.items():
        for sentence_id, sentence_prediction in enumerate(
            grammar.micro_evaluation_info
        ):
            data.append(
                {
                    "Grammar ID": grammar_id,
                    "Sentence ID": sentence_id,
                    "F1 Score": float(sentence_prediction["f1_score"]),
                    "Time": float(sentence_prediction["time"]),
                    "config": grammar.config,
                    "heuristics": grammar.config["heuristics"],
                    "learning_modes": grammar.config["learning_modes"],
                    "excluded_rolesets": grammar.config["excluded_rolesets"],
                }
            )

    df = pd.DataFrame(data)

    # Pivot the DataFrame to have Grammar IDs as columns and Sentence IDs as index
    pivoted_df = df.pivot(index="Sentence ID", columns="Grammar ID", values="F1 Score")

    # Batch the sentences
    batched_df = pivoted_df.groupby(np.arange(len(pivoted_df)) // batch_size).mean()

    # Calculate the cumulative average over the batches
    cumulative_avg_df = batched_df.expanding().mean()

    # Find the point where the change in cumulative F1 scores becomes very small
    cumulative_avg_diff = cumulative_avg_df.diff()
    plateau_batch = (cumulative_avg_diff.abs() < 0.0001).idxmax()

    # Calculate the average plateau point across all grammars
    avg_plateau_point = plateau_batch.mean()

    # Create an empty Figure object
    fig = go.Figure()

    # Add a trace for each grammar
    for grammar_id in cumulative_avg_df.columns:
        hover_text = [
            f"Grammar {row['Grammar ID']}<br>"
            f"F1-score: {row['F1 Score']:.3f}<br>"
            f"<br>Heuristics: {row['heuristics']}<br>"
            f"Learning Modes: {row['learning_modes']}<br>"
            f"Excluded Rolesets: {row['excluded_rolesets']}"
            for _, row in df[df["Grammar ID"] == grammar_id].iterrows()
        ]

        hover_template = "%{text}"

        fig.add_trace(
            go.Scatter(
                x=cumulative_avg_df.index,
                y=cumulative_avg_df[grammar_id],
                mode="lines+markers",
                name=f"Grammar {grammar_id}",
                text=hover_text,
                hovertemplate=hover_template,
            )
        )

    # Add a vertical line where the cumulative F1 scores are plateauing
    fig.add_shape(
        type="line",
        xref="x",
        x0=avg_plateau_point,
        x1=avg_plateau_point,
        yref="paper",
        y0=0,
        y1=1,
        line=dict(color="black", width=2),
        layer="below",
    )

    # Set the title and axis labels
    fig.update_layout(
        title=f"Cumulative Average F1-score for Each Batch of {batch_size} Sentences of Each Grammar",
        xaxis_title="Batch Number",
        yaxis_title="Cumulative Average F1-score",
        margin=dict(l=50, r=50, t=100, b=100),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.3)",
            title=dict(text="Configurations", font=dict(color="rgba(0, 0, 0, 0.7)")),
            font=dict(color="rgba(0, 0, 0, 0.7)"),
        ),
    )

    # Show the plot
    return fig


def plot_f1_scores(
    grammar_dict: Dict[int, Grammar],
    batch_size: int = 1,
    show_time: bool = False,
    plot_theme: str = "plotly",
):
    # Set default plotly theme to dark mode
    pio.templates.default = plot_theme

    # Transform the grammar_dict into a DataFrame
    data = []
    for grammar_id, grammar in grammar_dict.items():
        for sentence_id, sentence_prediction in enumerate(
            grammar.micro_evaluation_info
        ):
            data.append(
                {
                    "Grammar ID": grammar_id,
                    "Sentence ID": sentence_id,
                    "F1 Score": float(sentence_prediction["f1_score"]),
                    "Time": float(sentence_prediction["time"]),
                }
            )

    df = pd.DataFrame(data)

    # Pivot the DataFrame to have Grammar IDs as columns and Sentence IDs as index
    pivoted_df = df.pivot(index="Sentence ID", columns="Grammar ID", values="F1 Score")

    # Batch the sentences
    batched_df = pivoted_df.groupby(np.arange(len(pivoted_df)) // batch_size).mean()

    # Create an empty Figure object
    fig = go.Figure()

    # Add a trace for each grammar
    for grammar_id in batched_df.columns:
        fig.add_trace(
            go.Scatter(
                x=batched_df.index,
                y=batched_df[grammar_id],
                mode="lines+markers",
                name=f"Grammar {grammar_id}",
            )
        )

    # Set the title and axis labels
    fig.update_layout(
        title=f"F1-score for Each Batch of {batch_size} Sentences of Each Grammar",
        xaxis_title="Batch Number",
        yaxis_title="F1-score",
        width=1050,
        height=450,
        margin=dict(l=50, r=50, t=100, b=100),
        # Update legend settings for dark mode
        legend=dict(
            bgcolor="rgba(85, 85, 85, 0.3)",
            title=dict(text="Grammars", font=dict(color="white")),
            font=dict(color="white"),
        ),
    )

    # Show the plot
    fig.show()

    if show_time:
        # Pivot the DataFrame to have Grammar IDs as columns and Sentence IDs as index for normalized time
        pivoted_time_df = df.pivot(
            index="Sentence ID", columns="Grammar ID", values="Time"
        )

        # Batch the sentences
        batched_time_df = pivoted_time_df.groupby(
            np.arange(len(pivoted_time_df)) // batch_size
        ).mean()

        # Create an empty Figure object for normalized time
        fig_time = go.Figure()

        # Add a trace for each grammar
        for grammar_id in batched_time_df.columns:
            fig_time.add_trace(
                go.Scatter(
                    x=batched_time_df.index,
                    y=batched_time_df[grammar_id],
                    mode="lines+markers",
                    name=f"Grammar {grammar_id}",
                )
            )

        # Set the title and axis labels
        fig_time.update_layout(
            title=f"Time for Each Batch of {batch_size} Sentences of Each Grammar",
            xaxis_title="Batch Number",
            yaxis_title="Time",
            width=1050,
            height=450,
            margin=dict(l=50, r=50, t=100, b=100),
            # Update legend settings for dark mode
            legend=dict(
                bgcolor="rgba(85, 85, 85, 0.3)",
                title=dict(text="Grammars", font=dict(color="white")),
                font=dict(color="white"),
            ),
        )
        # Show the plot for normalized time
        fig_time.show()


def plot_sentence_f1_scores(
    grammar_dict: Dict[int, Grammar],
    sen_id: Tuple[int, int],
    display_mode: str = "grouped_grammars",
    orientation: str = "vertical",
    plot_theme: str = "plotly",
):
    pio.templates.default = plot_theme
    # Get the sentence prediction data for the specified sentence id
    sentence_data = []
    for grammar_id, grammar in grammar_dict.items():
        for info in grammar.micro_evaluation_info:
            if info["sen_id"] == sen_id:
                sentence_data.append(
                    {
                        "grammar_id": grammar_id,
                        "f1_score": float(info["f1_score"]),
                        "precision": float(info["precision"]),
                        "recall": float(info["recall"]),
                        "config": grammar.config,
                        "heuristics": grammar.config["heuristics"],
                        "learning_modes": grammar.config["learning_modes"],
                        "excluded_rolesets": grammar.config["excluded_rolesets"],
                    }
                )

    # Create a DataFrame
    df = pd.DataFrame(sentence_data)

    # Sort DataFrame by f1_score in descending order
    df = df.sort_values("grammar_id", ascending=False)

    # Create a bar chart
    fig = go.Figure()

    if display_mode == "grouped_grammars":
        for index, row in df.iterrows():
            hover_template = (
                f"Grammar {row['grammar_id']}<br>"
                f"Precision: {row['precision']:.3f}<br>"
                f"Recall: {row['recall']:.3f}<br>"
                f"F1-score: {row['f1_score']:.3f}<br>"
                f"<br>Heuristics: {row['heuristics']}<br>"
                f"Learning Modes: {row['learning_modes']}<br>"
                f"Excluded Rolesets: {row['excluded_rolesets']}"
            )

            if orientation == "vertical":
                fig.add_trace(
                    go.Bar(
                        x=[f"Grammar {row['grammar_id']}"],
                        y=[row["f1_score"]],
                        name=f"Grammar {row['grammar_id']}",
                        text=[f"{row['f1_score']:.3f}"],
                        textposition="auto",
                        hovertemplate=hover_template,
                    )
                )
            elif orientation == "horizontal":
                fig.add_trace(
                    go.Bar(
                        x=[row["f1_score"]],
                        y=[f"Grammar {row['grammar_id']}"],
                        name=f"Grammar {row['grammar_id']}",
                        text=[f"{row['f1_score']:.3f}"],
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
        title=f"Sentence {sen_id} F1-scores for Grammars",
        xaxis_title="Grammar ID" if orientation == "vertical" else "F1-score",
        yaxis_title="F1-score" if orientation == "vertical" else "Grammar ID",
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
