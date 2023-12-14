from typing import Dict
import re
from tabulate import tabulate
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import spacy
from spacy import displacy
from IPython.display import display, HTML
import pandas as pd
import pickle
import csv
import requests


@dataclass
class Frame:
    """
    This class represents a linguistic frame, which is a structure for organizing
    language understanding and production around specific concepts or scenarios.

    Attributes:
        frame_name (str): The name of the frame. This could be a semantic role label,
        a verb or other syntactic construction.

        lemma_name (str): The lemma (base form) of the word that this frame is
        associated with. This could be a verb, noun, or other part of speech.

        roles (List[Dict[str, any]]): A list of roles associated with this frame.
        Each role is represented as a dictionary with arbitrary keys and values.

        va_roles (List[Dict[str, any]]): An optional list of variant roles associated
        with this frame. Each role is also represented as a dictionary with
        arbitrary keys and values. This attribute defaults to None.

        va_frame_id (Optional[str]): An optional id of the variant frame associated
        with this frame. This attribute defaults to None.

        va_frame_name (Optional[str]): An optional name of the variant frame associated
        with this frame. This attribute defaults to None.
    """

    frame_name: str
    lemma_name: str
    roles: List[Dict[str, any]]
    va_roles: List[Dict[str, any]] = None
    va_frame_id: Optional[str] = None
    va_frame_name: Optional[str] = None


@dataclass
class Sentence:
    """
    This class represents a sentence which contains the id, string and frames.

    Attributes:
        sentence_id (Tuple[str, int]): A tuple that uniquely identifies the sentence.
        This typically includes a document identifier and the sentence's position
        within the document.

        sentence_string (str): The string representation of the sentence. This
        includes all the words in the sentence in their original order.

        frames (List[Frame]): A list of Frame objects associated with this sentence.
        Each frame captures a different aspect of the sentence's meaning, including
        the roles of different words and phrases within the sentence.
    """

    sentence_id: Tuple[str, int]
    sentence_string: str
    frames: List[Frame]


class VerbAtlasFrames:
    """
    This class represents the mapping between PropBank framesets and VerbAtlas frames,
    as well as additional information about VerbAtlas frames.

    Attributes:
        mapping (dict): A dictionary mapping PropBank senses to their corresponding
        VerbAtlas frame identifiers and role mappings.

        frame_info (dict): A dictionary mapping VerbAtlas frame identifiers to their
        associated frame names.

    Methods:
        _load_mapping(mapping_file_path): Load the mapping between PropBank senses and
        VerbAtlas frames from a tab-separated CSV file.

        _load_frame_info(frame_info_file_path): Load additional information about
        VerbAtlas frames from a tab-separated CSV file.

        get_verbatlas_mapping(propbank_sense): Get the VerbAtlas mapping for a given
        PropBank sense.

        _convert_propbank_role(propbank_role): Convert a PropBank role label to a
        simpler format, if applicable.

    Args:
        mapping_file_path (str): The path to the CSV file that contains the mapping
        between PropBank senses and VerbAtlas frames.

        frame_info_file_path (str): The path to the CSV file that contains additional
        information about VerbAtlas frames.
    """

    def __init__(self, mapping_file_path: str, frame_info_file_path: str):
        self.mapping = self._load_mapping(mapping_file_path)
        self.frame_info = self._load_frame_info(frame_info_file_path)

    def _load_mapping(self, mapping_file_path):
        mapping = {}
        with open(mapping_file_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter="\t")
            for row in csv_reader:
                if not row or ">" not in row[0]:
                    continue

                pb_sense, va_frame_id = row[0].split(">")

                # Extract the role mappings
                role_mappings = {}
                for role_map in row[1:]:
                    pb_role, va_role = role_map.split(">")
                    converted_pb_role = self._convert_propbank_role(pb_role)
                    role_mappings[converted_pb_role] = va_role

                mapping[pb_sense] = {
                    "va_frame_id": va_frame_id,
                    "role_mappings": role_mappings,
                }

        return mapping

    def _load_frame_info(self, frame_info_file_path):
        frame_info = {}
        with open(frame_info_file_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter="\t")
            for row in csv_reader:
                try:
                    frame_id = row[0]
                    if not frame_id.startswith("va:") or len(row) < 2:
                        continue

                    frame_number = int(frame_id[3:-1])

                    frame_info[frame_id] = row[1]
                except (ValueError, IndexError):
                    continue
        return frame_info

    def get_verbatlas_mapping(self, propbank_sense: str) -> Dict[str, str]:
        mapping = self.mapping.get(propbank_sense.lower(), {})
        va_frame_id = mapping.get("va_frame_id", "unknown")
        va_frame_name = self.frame_info.get(va_frame_id, "unknown")
        role_mappings = mapping.get("role_mappings", {})

        # Add additional role mappings for "V" and "FEE"
        role_mappings["V"] = "V"
        role_mappings["FEE"] = "FEE"

        return {
            "va_frame_id": va_frame_id,
            "va_frame_name": va_frame_name,
            "role_mappings": role_mappings,
        }

    def _convert_propbank_role(self, propbank_role: str) -> str:
        arg_match = re.match(r"ARG(\d)", propbank_role)
        if arg_match:
            return f"A{arg_match.group(1)}"

        am_match = re.match(r"AM-(\w+)", propbank_role)
        if am_match:
            return am_match.group(1)

        return propbank_role


class Corpus:
    """
    This class represents a corpus of sentences annotated with semantic role labels,
    together with associated frame semantic information from both PropBank and VerbAtlas.

    Attributes:
        sentences (Dict[Tuple[int, int], Sentence]): A dictionary mapping sentence
        identifiers to Sentence objects.

        source_file_index_map (Dict[str, int]): A mapping from source file names to
        integer indices.

        verb_atlas_frames (VerbAtlasFrames): An instance of the VerbAtlasFrames class.

    Methods:
        _parse_frame_roles(frame_roles): Parse the roles of a PropBank frame from a
        string representation.

        get_sentences(): Get a dictionary mapping sentence identifiers to sentence
        strings.

        get_gold_standard_frames(sentence_id): Get the list of PropBank frames for a
        given sentence, if available.

        add_verbatlas_mappings(): Add VerbAtlas frame information to each frame in the
        corpus.

        _source_file_to_index(source_file): Convert a source file name to an index.

        _load_corpus_data(file_path): Load the corpus data from a CSV file.

        get_sentence(sentence_id): Get a specific sentence from the corpus, given its
        identifier.

        get_sentences_with_va_frame(va_frame): Get the identifiers of all sentences
        in the corpus that include a specific VerbAtlas frame.

        save_corpus(file_path): Save the corpus to a file.

        load_corpus(file_path): Load a corpus from a file.

        visualize_sentence_table(sentence_id): Visualize the frame annotations of a
        specific sentence as a table.

        visualize_sentence_displacy(sentence_id): Visualize the frame annotations of
        a specific sentence using displaCy.

    Args:
        file_path (str): The path to a CSV file that contains the corpus data.

        verb_atlas_frames (VerbAtlasFrames): An instance of the VerbAtlasFrames class.
    """

    def __init__(self, file_path: str, verb_atlas_frames: VerbAtlasFrames):
        self.sentences: Dict[Tuple[int, int], Sentence] = {}
        self.source_file_index_map: Dict[str, int] = {}
        self.verb_atlas_frames = verb_atlas_frames
        self._load_corpus_data(file_path)

    def _parse_frame_roles(self, frame_roles: str) -> List[Dict[str, any]]:
        roles = []
        role_pattern = r"(\w+) \[(.+?)\]: (.+?)(?=;|$)"
        for match in re.finditer(role_pattern, frame_roles):
            role, indices, string = match.groups()
            indices = list(map(int, indices.split(" | ")))
            string = string.replace("~", ",")
            roles.append({"role": role, "indices": indices, "string": string})
        return roles

    def get_sentences(self) -> Dict[Tuple[int, int], str]:
        sentence_dict = {}
        for sentence_id, sentence in self.sentences.items():
            sentence_dict[sentence_id] = sentence.sentence_string
        return sentence_dict

    def get_gold_standard_frames(
        self, sentence_id: Tuple[int, int]
    ) -> Optional[List[Dict[str, any]]]:
        if sentence_id in self.sentences:
            return self.sentences[sentence_id].frames
        return None

    def add_verbatlas_mappings(self):
        for sentence_id, sentence in self.sentences.items():
            for frame in sentence.frames:
                propbank_id = frame.frame_name
                mapping = self.verb_atlas_frames.get_verbatlas_mapping(propbank_id)
                frame.va_frame_id = mapping["va_frame_id"]
                frame.va_frame_name = mapping["va_frame_name"]
                frame.va_roles = []

                for role in frame.roles:
                    pb_role = role["role"]
                    va_role = mapping["role_mappings"].get(
                        self.verb_atlas_frames._convert_propbank_role(pb_role),
                        "unknown",
                    )
                    frame.va_roles.append(
                        {
                            "role": va_role,
                            "indices": role["indices"],
                            "string": role["string"],
                        }
                    )

    # def add_verbatlas_mappings(self, base_url="http://localhost:5000"):
    #     for sentence_id, sentence in self.sentences.items():
    #         for frame in sentence.frames:
    #             # Assuming frame.lemma_name is the PropBank predicate sense like 'eat.01'
    #             propbank_id = frame.frame_name
    #             mapping = self._get_verbatlas_mapping(base_url, propbank_id)
    #             frame.va_frame_id = mapping['va_frame_id']
    #             frame.va_frame_name = mapping['va_frame_name']

    # def _get_verbatlas_mapping(self, base_url: str, propbank_sense: str) -> Dict[str, str]:
    #     url = f"{base_url}/api/verbatlas/align/sense"
    #     params = {"propbankSenseID": propbank_sense}
    #     response = requests.get(url, params=params)

    #     if response.status_code == 200:
    #         return response.json()
    #     else:
    #         raise Exception(f"Error: {response.status_code} - {response.text}")

    def _source_file_to_index(self, source_file: str) -> int:
        if source_file not in self.source_file_index_map:
            index = len(self.source_file_index_map)
            self.source_file_index_map[source_file] = index
        return self.source_file_index_map[source_file]

    def _load_corpus_data(self, file_path: str):
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1")

        df.replace("~", ",", inplace=True, regex=True)

        for _, row in df.iterrows():
            source_file = row["source_file"]
            source_file_index = self._source_file_to_index(source_file)
            sentence_id = row["sentence_id"]
            unique_sentence_id = (source_file_index, sentence_id)
            sentence_string = row["sentence_string"]
            frame_name = row["frame_name"]
            lemma_name = row["lemma_name"]
            frame_roles = self._parse_frame_roles(row["frame_roles"])

            # Get the corresponding VerbAtlas frame information
            mapping = self.verb_atlas_frames.get_verbatlas_mapping(frame_name)
            va_frame_id = mapping["va_frame_id"]
            va_frame_name = mapping["va_frame_name"]
            role_mappings = mapping["role_mappings"]

            # Convert the roles to va_roles using the role_mappings
            va_roles = []
            for role in frame_roles:
                converted_role = self.verb_atlas_frames._convert_propbank_role(
                    role["role"]
                )
                va_role = role_mappings.get(converted_role, "unknown")
                va_roles.append(
                    {
                        "role": va_role,
                        "indices": role["indices"],
                        "string": role["string"],
                    }
                )

            frame = Frame(
                frame_name,
                lemma_name,
                frame_roles,
                va_roles,
                va_frame_id,
                va_frame_name,
            )

            if unique_sentence_id in self.sentences:
                self.sentences[unique_sentence_id].frames.append(frame)
            else:
                sentence = Sentence(unique_sentence_id, sentence_string, [frame])
                self.sentences[unique_sentence_id] = sentence

    def get_sentence(self, sentence_id: Tuple[int, int]) -> Optional[Sentence]:
        return self.sentences.get(sentence_id)

    def get_sentences_with_va_frame(self, va_frame: str) -> List[Tuple[int, int]]:
        sentence_ids = []
        for sentence_id, sentence in self.sentences.items():
            for frame in sentence.frames:
                if frame.va_frame_name == va_frame:
                    sentence_ids.append(sentence_id)
                    break
        return sentence_ids

    def save_corpus(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self.sentences, f)
        print(f"Corpus saved to {file_path}")

    @classmethod
    def load_corpus(cls, file_path: str) -> "Corpus":
        with open(file_path, "rb") as f:
            sentences = pickle.load(f)
        corpus = cls.__new__(cls)
        corpus.sentences = sentences
        print(f"Corpus loaded from {file_path}")
        return corpus

    def visualize_sentence_table(self, sentence_id: Tuple[int, int]) -> None:
        sentence = self.get_sentence(sentence_id)
        print(f"Sentence: {sentence.sentence_string}\n")

        table_html = ""
        for frame in sentence.frames:
            table_headers = [f"ID: {sentence.sentence_id}", frame.frame_name]
            table_data = [[role["role"], role["string"]] for role in frame.roles]

            html_table = tabulate(table_data, headers=table_headers, tablefmt="html")
            table_html += f"{html_table}<br>"

        display(HTML(table_html))

    def visualize_sentence_displacy(self, sentence_id: Tuple[int, int]) -> None:
        sentence = self.get_sentence(sentence_id)

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence.sentence_string)

        # Create the table header with the total number of frames and sentence ID
        table_header = f"<h3>{len(sentence.frames)} frames for sentence ID: {sentence.sentence_id}</h3>"
        display(HTML(table_header))

        for frame in sentence.frames:
            # Generate displacy entities based on role_string
            entities = []
            for role in frame.roles:
                role_start = sentence.sentence_string.find(role["string"])
                role_end = role_start + len(role["string"])

                entities.append(
                    {
                        "start": role_start,
                        "end": role_end,
                        "label": role["role"],
                    }
                )

            render_data = [
                {
                    "text": doc.text,
                    "ents": entities,
                    "title": f"ID: {sentence.sentence_id} {frame.frame_name}",
                }
            ]

            # Define custom colors for each role type
            colors = {
                "ARG0": "#FCA311",
                "ARG1": "#2EC4B6",
                "ARG2": "#E63946",
                "ARG3": "#DD6E42",
                "ARG4": "#4EA8DE",
                "ARG5": "#57A773",
                "V": "#C8963E",
            }

            displacy_options = {
                "compact": True,
                "offset_x": 100,
                "distance": 100,
                "manual": True,
                "fine_grained": True,
                "colors": colors,
            }

            html = displacy.render(
                render_data,
                style="ent",
                manual=True,
                options=displacy_options,
                page=True,
                jupyter=False,
                minify=True,
            )
            display(HTML(html))
