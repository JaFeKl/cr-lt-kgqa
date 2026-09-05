import json
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()
langfuse = get_client()

DATASE_NAME = "CR-LT-KGQA"

# Create a dataset
langfuse.create_dataset(
    name=DATASE_NAME,  # required name
    # optional description
    description="A Knowledge Graph Question Answering Dataset Requiring Commonsense Reasoning and Long-Tail Knowledge",
    # optional metadata
    metadata={
        "authors": "Guo, Willis and Toroghi, Armin and Sanner, Scott",
        "date": "2024",
        "type": "benchmark",
        "paper": "arXiv:2403.01395",
    },
)

# load CoLoTa_qa.json dataset
with open("CoLoTa_qa.json", "r") as f:
    dataset_list = json.load(f)
    dataset = {item["id"]: item for item in dataset_list}

# load customized_dataset.jsonl
with open("customized_dataset.jsonl", encoding="utf-8") as file:
    dataset_custom = {
        item["id"]: item for line in file if line.strip() for item in [json.loads(line)]
    }


for id, entry in dataset_custom.items():
    standard_entry = dataset.get(id)

    langfuse.create_dataset_item(
        dataset_name=DATASE_NAME,
        # any python object or value, optional
        input=entry["question"],
        # any python object or value, optional
        expected_output=entry["a_entity"],
        # metadata, optional
        metadata={
            "kg_question_entitites": entry["q_entity"],
            "inference_rule": standard_entry["Inference Rule"],
            "kg_answer_triples": standard_entry["KG Triples"],
            "reasoining_steps": standard_entry["Reasoning Steps"],
            "reasoning_strategy": standard_entry["Reasoning Strategy"],
            "graph": entry["graph"],
        },
        id=id,
    )
