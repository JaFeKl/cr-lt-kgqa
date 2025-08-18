import os
import json
import requests
import re
from datetime import datetime
from abc import abstractmethod
from typing import List, Any, Dict, Tuple, Optional
from pydantic import BaseModel
from langchain_core.tools import tool
from multi_agent_lang_graph.ai_agents.agents.kg_agent.kgqa_datasets.graph import Graph
import pandas as pd
from multi_agent_lang_graph.ai_agents.agents.kg_agent.kgqa_datasets.datasets import (
    Dataset,
)
from multi_agent_lang_graph.ai_agents.agents.kg_agent.kgqa_datasets.wikidata import (
    Wikidata,
)


class CoLoTa(Dataset):
    def __init__(self, database_name: str = "CoLoTa"):
        super().__init__()
        self.data: pd.DataFrame = None
        self.wikidata = Wikidata()
        self.name = database_name
        # triples = self.wikidata.get_triples("Q27906395", True)
        # print(self.wikidata.verbalize_triples(triples[0]))
        # triples = self.wikidata.verbalize_triples(triples)
        # print(triples)
        # self.cache_file_path = cache_file_path

    def load_dataset(self, dataset_path, cache_file_path=None):
        super().load_dataset(dataset_path)
        # reformormat "KG Triples" to a list of triples
        cache = None

        if cache_file_path is not None:
            cache = self.load_cached_data(cache_file_path)

        for entry in self.dataset:
            triples = entry.get("KG Triples", "")

            # Regex to extract triples
            pattern = r"\d+- \(([^,]+), ([^,]+), ([^)]+)\)"
            matches = re.findall(pattern, triples)

            # Convert matches to a list of tuples
            triples = [
                (match[0].strip(), match[1].strip(), match[2].strip())
                for match in matches
            ]
            entry["KG Triples"] = triples

            if cache is not None:
                # check if the entry is already in the cache
                cache_entry = cache[cache["id"] == entry["id"]]
                if not cache_entry.empty:
                    entry["graph"] = cache_entry.iloc[0]["graph"]
                else:
                    entry["graph"] = []

    def dump_to_mongo(
        self,
        collection_name: str = "dataset",
        overwrite: bool = True,
        graph_date: Optional[datetime] = None,
    ):
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Please load the dataset first.")

        for entry in self.dataset:
            if graph_date is not None:
                if entry.get("graph") is not None:
                    entry["graph_date"] = graph_date
            matched_entries = self.client.match_entry(
                self.name, collection_name, {"id": entry["id"]}
            )
            if matched_entries is None or matched_entries == []:
                print(f"Entry with ID {entry['id']} not found. Creating a new entry.")
                self.client.create_entry(
                    data=entry,
                    database_name=self.name,
                    collection_name=collection_name,
                )

            elif len(matched_entries) == 1:
                if overwrite is True:
                    print(f"Entry with ID {entry['id']} found. Updating the entry.")
                    self.client.create_or_update_entry(
                        data=entry,
                        database_name=self.name,
                        collection_name=collection_name,
                    )
                else:
                    print(f"Entry with ID {entry['id']} already exists. Skipping.")
                    continue
            else:
                ValueError(
                    f"Multiple entries found for ID {entry['id']}. Please check the database."
                )

    def load_cached_data(self, cache_file_path: str = None):
        """Load the data from the file and store it in a pandas DataFrame"""
        data = []
        with open(cache_file_path, "r", encoding="utf-8") as file:
            for line in file:
                line_data = json.loads(line)
                data.append(line_data)
        return pd.DataFrame(data)

    def prepare_question(self, _id: str) -> str:
        db_entry = self.mongo_db.client.get_entry_by_id(
            _id,
            "dataset",
            self.name,
        )
        self.create_graph(db_entry["graph"])
        return db_entry["query"]

    def create_graph(self, graph: List):
        self.graph.clear()
        for triple in graph:
            relation = triple[2].get("relation")
            metadata = {k: v for k, v in triple[2].items() if k != "relation"}
            self.graph.add_triple(triple[0], relation, triple[1], **metadata)

    def _parse_triples(self, triples: str) -> List:
        parsed_triples = []

        for triple in triples:
            # Handle cases with metadata (e.g., "{...}")
            if "{" in triple:
                # Extract the main triple and metadata
                main_part = re.search(r"\(\((.*?)\),", triple)
                metadata_part = re.search(r"\{(.*?)\}", triple)

                if main_part:
                    # Split the main part into subject, predicate, and object
                    parts = main_part.group(1).split(
                        ", ", 2
                    )  # Split into at most 3 parts
                    if len(parts) == 3:
                        subject, predicate, obj = parts
                        metadata = {}
                        if metadata_part:
                            # Parse metadata into a dictionary
                            metadata = dict(
                                item.split(": ", 1)
                                for item in metadata_part.group(1).split("; ")
                                if ": " in item
                            )
                        # Add edge with metadata
                        metadata["relation"] = predicate
                        parsed_triples.append((subject, obj, metadata))
            else:
                # Handle regular triples
                triple = triple.strip("()")
                parts = triple.split(", ", 2)  # Split into at most 3 parts
                if len(parts) == 3:
                    subject, predicate, obj = parts
                    # Add edge without metadata
                    parsed_triples.append((subject, obj, {"relation": predicate}))
        return parsed_triples

    def _extract_wikidata_subgraph(
        self, kg_entities: Dict, date: datetime
    ) -> Tuple[List, Dict]:
        subgraph = []
        qid_dict = {}
        for entity, qid in kg_entities.items():
            triples_str = []
            is_label = False
            if not qid:
                if qid_dict.get(entity):
                    kg_entities[entity] = qid_dict[entity]
                    qid = qid_dict[entity]
                else:
                    is_label = True
                    qid = entity
            try:
                # get triples from wikidata
                revisions = self.wikidata.get_revision_history(qid, is_label=is_label)
                if revisions is None:
                    print(f"No revisions found for {qid}")
                    continue

                # get the latest revision before the date
                revision = self.wikidata.get_last_revision_before_date(revisions, date)
                if revision is None:
                    print(f"No revision found for {qid} before {date}")
                    continue

                triples = self.wikidata.get_triples(
                    qid, is_label=is_label, revision_id=revision
                )
                qid_dict = qid_dict | triples[1]  # merge the qid dict to the qid_dict

                verbalized_triples = self.wikidata.verbalize_triples(triples[0])

                triples_str = self._parse_triples(verbalized_triples)
                subgraph.extend(triples_str)
            except requests.exceptions.RequestException as e:
                print(f"Failed to retrieve triples for {entity} ({qid}): {e}")
            # except Exception as e:
            #     print(f"Failed to retrieve triples for {entity} ({qid}): {e}")

        return subgraph, kg_entities

    def create_wikidata_cache(self, dataset_path: str, cache_path: str, date: datetime):
        """Create a jsonl file which include wikidata subgraphs for each question"""
        last_id = 0
        data_list = []
        # check if the cache file already exists
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                # load cache file and check until which id the cache is created
                data_list = [json.loads(line) for line in cache_file]
                last_id = len(data_list)
                if last_id > 0:
                    print(f"Cache file already exists. Last ID: {last_id}")
        except FileNotFoundError:
            print(f"Cache file not found. Creating a new one: {cache_path}")
            pass

        def save_jsonl(data_list, cache_path):
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                for entry in data_list:
                    cache_file.write(json.dumps(entry) + "\n")

        # open json of the dataset
        with open(dataset_path, "r", encoding="utf-8") as file:
            self.dataset = json.load(file)

        # create or append a jsonl cache file
        required_triples = []

        # iterate over the dataset starting from the last id
        for entry in self.dataset[last_id:]:
            print("Processing entry:", entry["id"])
            kg_entities = {}
            for reasoning_step in entry["Reasoning Steps"]:
                triple = reasoning_step.get("facts used in this step")

                if not triple:
                    continue
                triple = triple.strip("()").split(", ")
                required_triples.append(triple)

                qid = None
                if triple[0] in entry["KG Entities"]:
                    qid = entry["KG Entities"][triple[0]]

                kg_entities[triple[0]] = qid

            subgraph, kg_entities = self._extract_wikidata_subgraph(kg_entities, date)
            # check if the required triples are part of the graph
            for triple in required_triples:
                # check if the triple is in the subgraph, the triple is in the format [head, realtion, tail]
                # but the subgraph has entries in the format (head, tail, metadata)
                # need to check first if the head and tail are in the subgraph and then check if the relation is in the metadata
                if not any(t[0] == triple[0] and t[1] == triple[2] for t in subgraph):
                    print(f"Caution: Triple {triple} is not part of the graph")
                    continue

            cache_entry = {
                "id": entry["id"],
                "question": entry["query"],
                "q_entity": kg_entities,
                "a_entity": entry["answer"],
                "graph": subgraph,
            }
            data_list.append(cache_entry)
            save_jsonl(data_list, cache_path)

        # write the data_list to a jsonl file

        # with open("wikidata_cache.jsonl", "w", encoding="utf-8") as cache_file:
        #     for entry in data_list:
        #     question_id = entry["id"]
        #     triples = entry.get("KG Triples", "")
        #     if triples:
        #         triples_list = []
        #         for triple in triples.split(", "):
        #         triple_parts = triple.split("- ")[1].strip("()").split(", ")
        #         if len(triple_parts) == 3:
        #             triples_list.append((triple_parts[0], triple_parts[1], triple_parts[2]))
        #         subgraph = self.wikidata.get_subgraph(triples_list)
        #         cache_entry = {
        #         "id": question_id,
        #         "subgraph": subgraph
        #         }
        #         cache_file.write(json.dumps(cache_entry) + "\n")

    def get_questions_per_hop(self, hop: int) -> List[str]:
        """Get a list of question IDs that have a specific number of hops"""
        if self.data is None:
            self.load_cached_data()
        # print(self.data["q_entity"].apply(len))
        questions = self.data[
            self.data["q_entity"].apply(lambda x: len(x) if isinstance(x, dict) else 0)
            == hop
        ]["id"].tolist()
        return questions
        # return questions

    def get_question(self, question_id: str) -> str:
        question_data = self.data[self.data["id"] == question_id]
        if not question_data.empty:
            question_data = question_data.iloc[0]
            return question_data["question"]
        else:
            raise ValueError(f"Question ID {question_id} not found in the dataset.")

    def set_result(self, question_id: str, result: Dict):
        if self.results is None:
            self.results = pd.DataFrame(columns=["id", "result"])

        # get the dataset expected result
        entry = self.data[self.data["id"] == question_id]
        label = entry.iloc[0]["a_entity"] if not entry.empty else None

        self.results = self.results._append(
            {
                "dataset_entry_id": question_id,
                "result": result["finalAnswer"]["answer"],
                "label": label,
                "explanation": result["finalAnswer"]["explanation"],
                "reasoning_steps": result["reasoningSteps"],
            },
            ignore_index=True,
        )


# dataset = CoLoTa(database_name="2025_08_15_qwq30B")
# dataset.create_wikidata_cache(
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/kgqa_datasets/cr_lt_kgqa/CoLoTa_qa.json",
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/kgqa_datasets/cr_lt_kgqa/2025_07_28_CR-LT-QA-Wikidata-Cache_2022_11_01.jsonl",
#     datetime(2025, 8, 20),
# )

# dataset.load_dataset(
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/kgqa_datasets/cr_lt_kgqa/CoLoTa_qa.json",
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/kgqa_datasets/cr_lt_kgqa/customized_dataset.jsonl",
# )
# dataset.dump_to_mongo(
#     overwrite=True,
# )
# dataset.create_json_from_jsonl(
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/datasets/cr_lt_kgqa/2025_07_09_CR-LT-QA-Wikidata-Cache_2022_11_01.jsonl",
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/datasets/cr_lt_kgqa/2025_07_09_dataset.json",
# )

# dataset.load_dataset(
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/datasets/cr_lt_kgqa/2025_07_09_CR-LT-QA-Wikidata-Cache_2022_11_01.jsonl"
# )

# dataset.create_wikidata_cache(
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/datasets/cr_lt_kgqa/CR-LT-QA.json",
#     "/home/sim/multi_agent/multi_agent_lang_graph/ai_agents/agents/kg_agent/datasets/cr_lt_kgqa/2025_07_09_CR-LT-QA-Wikidata-Cache_2022_11_01.jsonl",
#     datetime(2022, 11, 1),
# )
