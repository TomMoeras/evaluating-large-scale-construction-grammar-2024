import plotly.express as px
import plotly
from json import JSONEncoder
import json
from flask import Flask, render_template, request, jsonify, Response
import sys  # nopep8
from pathlib import Path  # nopep8

# Add the 'src' directory to sys.path
src_path = Path(__file__).resolve().parent.parent  # nopep8
sys.path.append(str(src_path))  # nopep8
from grammar_dict_analysis.visualization import *
from grammar_dict_analysis.load_grammar_dict_pickle import (
    load_grammar_dict_from_pickle,
    GrammarDictUnpickler,
)
from grammar_corpus import (
    Grammar,
    GrammarManager,
    Prediction,
    Corpus,
    Frame,
    Sentence,
    VerbAtlasFrames,
)
from grammar_dict_analysis.performance import *
from tabulate import tabulate
from math import isnan

grammar_dict_pickle_dir = Path(__file__).parent.parent.parent / "grammar_dict_pickle"
pickle_file_path = (
    grammar_dict_pickle_dir / "exp_full_train_4000_predictions_grammar_objects.pkl"
)


class PredictionEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Prediction):
            return {
                "sentence": str(o.sentence),
                "micro_evaluation_data": self._replace_nan_with_zero(
                    o.micro_evaluation_data
                ),
                "grammar_predicted_frames": o.grammar_predicted_frames,
            }
        elif isinstance(o, Frame):
            return {
                "frame_name": o.frame_name,
                "lemma_name": o.lemma_name,
                "roles": o.roles,
            }
        elif isinstance(o, Grammar):
            return {
                "grammar_id": o.grammar_id,
                "grammar_size": o.grammar_size,
                "config": o.config,
                "evaluation_scores": self._replace_nan_with_zero(o.evaluation_scores),
                "frame_roleset_performance": o.frame_roleset_performance,
                "micro_evaluation_info": self._replace_nan_with_zero(
                    o.micro_evaluation_info
                ),
            }
        return super(PredictionEncoder, self).default(o)

    def _replace_nan_with_zero(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._replace_nan_with_zero(value)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                data[i] = self._replace_nan_with_zero(value)
        elif isinstance(data, float) and isnan(data):
            return 0
        return data


grammar_dict, corpus = load_grammar_dict_from_pickle(pickle_file_path)

manager = GrammarManager(corpus)
manager.grammar_dict = grammar_dict


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_sentences")
def get_sentences():
    # key is sentence id (tuple) and value is sentence string
    sentences = corpus.get_sentences()

    # Convert tuple keys to strings
    sentences_dict = {str(k): v for k, v in sentences.items()}

    return jsonify(sentences_dict)


@app.route("/performance_parameters/<sentence_id>")
def performance_parameters_sentence(sentence_id):
    sentence_id_tuple = tuple(map(int, sentence_id.strip("()").split(",")))

    # Rename the variable to avoid conflict with the function name
    perf_parameters = performance_parameters_optimized(
        grammar_dict,
        full_table=False,
        level=1,
        metric="f1_score",
        sen_id=sentence_id_tuple,
    )

    performance_table_html = tabulate(perf_parameters, tablefmt="html")

    return performance_table_html


@app.route("/f1_score_plot/<sentence_id>")
def f1_score_plot(sentence_id):
    sentence_id_tuple = tuple(map(int, sentence_id.strip("()").split(",")))
    fig = plot_sentence_f1_scores(
        grammar_dict, sentence_id_tuple, orientation="horizontal"
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route("/time_sentence_plot/<sentence_id>")
def time_sentence_plot(sentence_id):
    sentence_id_tuple = tuple(map(int, sentence_id.strip("()").split(",")))
    fig = plot_sentence_time(grammar_dict, sentence_id_tuple, orientation="horizontal")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route("/grammar_info/<sentence_id>")
def grammar_info(sentence_id):
    print(f"Received sentence_id: {sentence_id}")
    sentence_id_tuple = tuple(map(int, sentence_id.strip("()").split(",")))
    print(f"Converted sentence id: {sentence_id_tuple}")
    print(f"Type of sentence id: {type(sentence_id_tuple)}")

    sorted_summaries = manager.get_sorted_grammars_for_sentence(sentence_id_tuple)

    if len(sorted_summaries) == 0:
        print("No grammar summaries found")

    if len(sorted_summaries) > 0:
        print(f"Found {len(sorted_summaries)} grammar summaries")

    response = json.dumps(sorted_summaries, cls=PredictionEncoder)
    return Response(response, mimetype="application/json")


@app.route("/visualize_prediction/<grammar_id>/<sentence_id>")
def visualize_prediction(grammar_id, sentence_id):
    print(f"Received grammar_id: {grammar_id}")
    grammar = manager.get_grammar(int(grammar_id))
    print(f"Received sentence_id: {sentence_id}")
    sentence_id_tuple = tuple(map(int, sentence_id.strip("()").split(",")))
    print(f"Converted sentence id: {sentence_id_tuple}")

    # Get the FramePrediction object for the given sentence_id
    prediction = grammar.get_prediction(sentence_id_tuple)
    print(prediction)

    # Check if the prediction is None or empty
    if not prediction or not prediction.grammar_predicted_frames:
        sentence = manager.corpus.sentences.get(sentence_id_tuple)
        prediction = Prediction(sentence)
        print(prediction)

    visualization_data = prediction.visualize_prediction()
    table_html = visualization_data.get("table_html", "")
    displacy_html = visualization_data.get("displacy_html", "")
    response_data = {"table_html": table_html, "displacy_html": displacy_html}

    response = json.dumps(response_data, cls=PredictionEncoder)
    return Response(response, mimetype="application/json")


@app.route("/get_micro_evaluation_info/<grammar_id>/<sentence_id>")
def get_micro_evaluation_info(grammar_id, sentence_id):
    grammar = manager.get_grammar(int(grammar_id))
    sentence_id_tuple = tuple(map(int, sentence_id.strip("()").split(",")))

    micro_eval_info = grammar.get_micro_evaluation_info(sentence_id_tuple)

    if micro_eval_info is None:
        response_data = {"message": "No micro evaluation info found."}
    else:
        response_data = micro_eval_info

    frames_info = grammar.get_grammar_predicted_frames(sentence_id_tuple)

    response_data = {"micro_eval_info": response_data, "frames_info": frames_info}

    response = json.dumps(response_data, cls=PredictionEncoder)
    return Response(response, mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True)
