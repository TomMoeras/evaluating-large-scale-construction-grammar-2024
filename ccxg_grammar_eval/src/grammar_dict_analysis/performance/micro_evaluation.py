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
from grammar_corpus import Grammar, GrammarManager, Prediction, Corpus, Frame, Sentence, VerbAtlasFrames


def correlation_analysis(grammar_dict: Dict[int, Grammar]) -> pd.DataFrame:
    # Create an empty DataFrame to store the micro_evaluation_info for all grammars
    all_data = pd.DataFrame()

    # Iterate through each grammar in the grammar_dict
    for grammar_id, grammar in grammar_dict.items():
        # Convert the time_info list of dictionaries into a DataFrame
        grammar_df = pd.DataFrame(grammar.micro_evaluation_info)

        # Add a column for the grammar_id
        grammar_df['grammar_id'] = grammar_id

        # Append the grammar_df to the all_data DataFrame
        all_data = all_data.append(grammar_df, ignore_index=True)

    # Calculate the correlations between the variables of interest
    predictive_vars = ['nr_tokens', 'nr_frames', 'nr_aux',
                       'nr_roles', 'nr_core', 'nr_argm', 'same_lemma', 'same_verb']
    target_vars = ['f1_score', 'time']
    correlation_matrix = all_data[predictive_vars + target_vars].corr()

    # Filter the correlation matrix to only show correlations between predictive and target variables
    correlation_matrix = correlation_matrix.loc[predictive_vars, target_vars]

    return correlation_matrix


def interpret_correlation_matrix(correlation_matrix: pd.DataFrame) -> List[str]:
    interpretation = []

    # Iterate through each row (variable) in the correlation matrix
    for index, row in correlation_matrix.iterrows():
        # Skip the correlations with 'f1_score' and 'time' themselves
        if index in ['f1_score', 'time']:
            continue

        # Interpret the correlation with f1_score
        f1_score_corr = row['f1_score']
        if abs(f1_score_corr) > 0.7:
            strength = 'strong'
        elif abs(f1_score_corr) > 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'

        interpretation.append(
            f"The correlation between {index} and f1_score is {strength} ({f1_score_corr:.2f}).")

        # Interpret the correlation with time
        time_corr = row['time']
        if abs(time_corr) > 0.7:
            strength = 'strong'
        elif abs(time_corr) > 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'

        interpretation.append(
            f"The correlation between {index} and time is {strength} ({time_corr:.2f}).")

    return interpretation


def time_analysis(grammar_dict: Dict[int, Grammar]) -> pd.DataFrame:
    time_data = []

    for grammar_id, grammar in grammar_dict.items():
        for info in grammar.micro_evaluation_info:
            time_data.append({
                'grammar_id': grammar_id,
                'sen_id': info['sen_id'],
                'time': info['time']
            })

    time_df = pd.DataFrame(time_data)

    # Calculate the mean time for each grammar
    time_stats = time_df.groupby('grammar_id').agg(
        {'time': ['mean']}
    ).reset_index()

    time_stats.columns = ['grammar_id', 'mean_time']

    # Sort the time stats by mean time
    time_stats = time_stats.sort_values(by='mean_time', ascending=True)

    return time_stats
