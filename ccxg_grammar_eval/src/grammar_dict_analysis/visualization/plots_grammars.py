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
src_path = Path(__file__).resolve().parent.parent.parent  # nopep8
sys.path.append(str(src_path))  # nopep8
# Import the necessary module components
from grammar_corpus.corpus import Corpus, Frame, Sentence, VerbAtlasFrames
from grammar_corpus.grammar_objects import Grammar, GrammarManager, Prediction
from grammar_dict_analysis.performance import *


def frame_scores_heatmap(grammar_dict: Dict[int, Grammar]):
    # Get the ordered labels from the dendrogram
    ordered_labels = cluster_frames(grammar_dict)

    # Create a DataFrame from the grammar_dict
    data = []
    for grammar_id, grammar in grammar_dict.items():
        for frame_name, frame_prediction in grammar.frame_predictions.items():
            data.append({
                'Grammar ID': grammar_id,
                'Frame Name': frame_name,
                'Precision': float(frame_prediction['precision']),
                'Recall': float(frame_prediction['recall']),
                'F1 Score': float(frame_prediction['f1_score']),
                'Support': float(frame_prediction['support'])
            })

    df = pd.DataFrame(data)

    # Calculate the average scores for each frame
    avg_df = df.groupby('Frame Name').agg(
        {'Precision': 'mean', 'Recall': 'mean', 'F1 Score': 'mean', 'Support': 'mean'})

    # Normalize the 'Support' column using min-max normalization
    avg_df['Support_normalized'] = (avg_df['Support'] - avg_df['Support'].min()) / \
        (avg_df['Support'].max() - avg_df['Support'].min())

    # Reorder the avg_df based on the ordered_labels
    avg_df = avg_df.loc[ordered_labels]

    # Create a heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=avg_df[['Precision', 'Recall', 'F1 Score',
                  'Support_normalized']].values,
        x=['Precision', 'Recall', 'F1 Score', 'Support_normalized'],
        y=avg_df.index,
        colorscale='Electric',
        zmin=0,
        zmax=1,
        text=avg_df[['Precision', 'Recall',
                     'F1 Score', 'Support']].values.round(2),
        hovertemplate='%{y}<br>%{x}: %{text}',
        showscale=True
    ))

    fig.update_layout(
        title='Average Frame Scores Heatmap',
        xaxis_title='Metric',
        yaxis_title='Frame Name'
    )

    fig.show()


def performance_parameters_heatmap(result_table: pd.DataFrame):
    # Get the ordered labels from the dendrogram
    ordered_labels = cluster_performance_parameters(result_table)

    # Reorder the result_table based on the ordered_labels
    result_table = result_table.loc[ordered_labels]

    # Create a heatmap using plotly
    fig = go.Figure()

    # Loop through the metrics in the result_table and add a Heatmap trace for each
    for i, metric in enumerate(result_table.columns.get_level_values(1).unique()):
        trace_df = result_table.xs(metric, axis=1, level=1)

        fig.add_trace(go.Heatmap(
            z=trace_df.values,
            x=[metric],
            y=trace_df.index,
            colorscale='Electric',
            zmin=trace_df.values.min(),
            zmax=trace_df.values.max(),
            text=trace_df.values.round(3),
            hovertemplate=f'%{{y}}<br>{metric}: %{{text}}',
            showscale=False,
            xaxis=f'x{i+1}',
            yaxis='y1'
        ))

    # Set the layout for the heatmap
    fig.update_layout(
        title='Performance Parameters Heatmap',
        yaxis_title='Parameter',
        xaxis=dict(domain=[0, 0.3], title='Mean'),
        xaxis2=dict(domain=[0.35, 0.65], title='Std'),
        xaxis3=dict(domain=[0.7, 1], title='Diff'),
        yaxis1=dict(title='Parameter'),
        grid=dict(columns=3, rows=1, pattern='independent')
    )

    fig.show()


def visualize_correlation_analysis(correlation_matrix: pd.DataFrame):
    # Set the default Plotly theme to dark mode
    pio.templates.default = 'plotly_dark'

    # Create a heatmap using Plotly
    fig = go.Figure(go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Electric',
        zmin=-1,
        zmax=1,
        text=correlation_matrix.values.round(2),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{text}',
    ))

    # Set the layout for the heatmap
    fig.update_layout(
        title='Correlation Analysis Heatmap',
        xaxis_title='Target Variables',
        yaxis_title='Predictive Variables',
        width=800,
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    # Show the plot
    fig.show()
