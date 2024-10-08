import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging
import pandas as pd
from grammar_corpus import GrammarManager

def extract_f1_scores(grammar_dict: Dict[int, 'Grammar']) -> Dict[int, List[float]]:
    logging.info("Extracting F1 scores from grammar_dict")
    f1_scores = {}
    all_sentences = set()
    
    # First, collect all valid sentences and their F1 scores
    for grammar_id, grammar in grammar_dict.items():
        f1_scores[grammar_id] = {}
        for info in grammar.micro_evaluation_info:
            sen_id = info['sen_id']
            f1_score = info['f1_score']
            if f1_score != 'nan' and not pd.isna(f1_score):
                try:
                    f1_scores[grammar_id][sen_id] = float(f1_score)
                    all_sentences.add(sen_id)
                except ValueError:
                    logging.warning(f"Invalid F1 score for grammar {grammar_id}, sentence {sen_id}: {f1_score}")
    
    # Keep only sentences where all grammars made a prediction
    valid_sentences = [sen for sen in all_sentences if all(sen in f1_scores[g] for g in f1_scores)]
    
    # Create the final f1_scores dictionary
    final_f1_scores = {g: [f1_scores[g][sen] for sen in valid_sentences] for g in f1_scores}
    
    all_scores = [score for scores in final_f1_scores.values() for score in scores]
    if all_scores:
        logging.info(f"Extracted a total of {len(all_scores)} valid F1 scores from {len(grammar_dict)} grammars")
        logging.debug(f"Overall F1 score statistics: Mean: {np.mean(all_scores):.4f}, Std: {np.std(all_scores):.4f}")
        logging.debug(f"F1 score range: Min: {min(all_scores):.4f}, Max: {max(all_scores):.4f}")
    else:
        logging.warning("No valid F1 scores extracted from any grammar")
    
    return final_f1_scores

def bootstrap_sample(data: List[float], num_samples: int, sample_size: int) -> np.ndarray:
    logging.debug(f"Creating bootstrap sample: num_samples={num_samples}, sample_size={sample_size}")
    return np.random.choice(data, (num_samples, sample_size), replace=True)

def calculate_ci(bootstrap_samples: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    logging.debug(f"Calculating confidence interval: confidence_level={confidence_level}")
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    ci = np.percentile(bootstrap_samples, [lower_percentile, upper_percentile])
    logging.debug(f"Confidence interval: {ci}")
    return ci

def bootstrap_analysis(grammar_dict: Dict[int, 'Grammar'], num_bootstrap_samples: int = 10000, confidence_level: float = 0.95) -> Dict[int, Dict[str, float]]:
    logging.info(f"Performing bootstrap analysis: num_bootstrap_samples={num_bootstrap_samples}, confidence_level={confidence_level}")
    f1_scores = extract_f1_scores(grammar_dict)
    results = {}

    for grammar_id, scores in f1_scores.items():
        if scores:
            logging.debug(f"Analyzing grammar {grammar_id}")
            bootstrap_means = np.mean(bootstrap_sample(scores, num_bootstrap_samples, len(scores)), axis=1)
            ci_lower, ci_upper = calculate_ci(bootstrap_means, confidence_level)
            results[grammar_id] = {
                'mean': np.mean(scores),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            logging.debug(f"Grammar {grammar_id} results: mean={results[grammar_id]['mean']:.4f}, CI=({ci_lower:.4f}, {ci_upper:.4f})")
        else:
            logging.warning(f"No valid scores for grammar {grammar_id}")

    return results

def compare_grammars_bootstrap(grammar_dict: Dict[int, 'Grammar'], num_bootstrap_samples: int = 10000, confidence_level: float = 0.95) -> Dict[int, Dict[str, float]]:
    logging.info("Comparing grammars using bootstrap analysis")
    results = bootstrap_analysis(grammar_dict, num_bootstrap_samples, confidence_level)
    
    logging.info(f"Bootstrap Analysis Results (Confidence Level: {confidence_level*100}%)")
    logging.info("Grammar ID | Mean F1 Score | Confidence Interval")
    logging.info("-" * 50)
    for grammar_id, stats in results.items():
        logging.info(f"{grammar_id:10d} | {stats['mean']:.4f} | ({stats['ci_lower']:.4f}, {stats['ci_upper']:.4f})")

    # Compare confidence intervals
    grammar_ids = list(results.keys())
    for i in range(len(grammar_ids)):
        for j in range(i+1, len(grammar_ids)):
            id1, id2 = grammar_ids[i], grammar_ids[j]
            if (results[id1]['ci_lower'] > results[id2]['ci_upper'] or 
                results[id2]['ci_lower'] > results[id1]['ci_upper']):
                logging.info(f"Statistically significant difference between Grammar {id1} and Grammar {id2}")
            else:
                logging.info(f"No statistically significant difference between Grammar {id1} and Grammar {id2}")

    return results

# Example usage
# compare_grammars_bootstrap(grammar_dict)