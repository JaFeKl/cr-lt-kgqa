import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Optional, Dict

from experiment_handler import ExperimentHandler
from experiment_handler.storage import MongoDBExperimentStorage, ExperimentSeries
from experiment_handler.models import Status

# EXPERIMENT_DB = "2025_08_15_qwq30B"
# EXPERIMENT_SERIES = "689edebe05b36806f734ddac"
# DATASET_DB = "2025_08_15_qwq30B"


handler = ExperimentHandler("Visualization")
handler.add_storage(
    MongoDBExperimentStorage(
        "admin",
        os.environ.get("MONGO_HOST"),
        os.environ.get("MONGO_PORT"),
        os.environ.get("MONGO_USERNAME"),
        os.environ.get("MONGO_PASSWORD"),
    )
)


# Load the dataset JSON
@st.cache_data
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load the results CSV
@st.cache_data
def load_results(csv_path):
    return pd.read_csv(csv_path)


def get_corla_answer_from_dataset_entry(dataset_id: str, database_name: str) -> bool:
    """Retrieve the answer from the dataset entry for a specific dataset ID."""
    entry = handler.storage.get(
        object_id=dataset_id, collection="dataset", database=database_name
    )
    return entry["answer"]


def get_corla_id_from_dataset_entry(dataset_id: str, database_name: str) -> str:
    """Retrieve the ID from the dataset entry for a specific dataset ID."""
    entry = handler.storage.get(
        object_id=dataset_id, collection="dataset", database=database_name
    )
    return entry["id"]


def get_run_results(dataset_id: str, series: ExperimentSeries) -> List[Dict]:
    """Retrieve results for a specific dataset ID from the experiment series."""
    results = []
    for experiment_id in series.experiment_ids:
        experiment = handler.storage.get_experiment(experiment_id)
        config_id = experiment.config_id
        config = handler.storage.get_configuration(config_id)
        if config.parameters["dataset_entry_id"] == dataset_id:
            for run_id in experiment.run_ids:
                run = handler.storage.get_run(run_id)
                results.append(run.results)
    return results


def convert_run_results_to_dataframe(
    series: ExperimentSeries, run_index: int, database_name: str
) -> pd.DataFrame:

    # create pandas Series with two columns: experiment_id, run results
    data = {
        "id": [],
        "ground_truth": [],
        "answer": [],
    }

    for experiment_id in series.experiment_ids:
        experiment = handler.storage.get_experiment(experiment_id)
        run_id = experiment.run_ids[run_index]
        run = handler.storage.get_run(run_id)
        if run.status != Status.COMPLETED:
            continue
        # get dataset_id
        data["id"].append(
            get_corla_id_from_dataset_entry(run.results["dataset_id"], database_name)
        )
        data["ground_truth"].append(
            get_corla_answer_from_dataset_entry(
                run.results["dataset_id"], database_name
            )
        )
        answer = run.results.get("answer", {}).get("answer")
        if answer is None:
            data["answer"].append(pd.NA)
        else:
            data["answer"].append(answer)

    df = pd.DataFrame(data)
    df["TP"] = ((df["ground_truth"] == True) & (df["answer"] == True)).astype(int)
    df["TN"] = ((df["ground_truth"] == False) & (df["answer"] == False)).astype(int)
    df["FP"] = ((df["ground_truth"] == False) & (df["answer"] == True)).astype(int)
    df["FN"] = ((df["ground_truth"] == True) & (df["answer"] == False)).astype(int)
    df["Undecisive"] = df["answer"].isna().astype(int)

    df.to_csv("run_results.csv", index=False, na_rep="None")
    return df


def calculate_metrics_per_run(
    series: ExperimentSeries, run_index: int, database_name: str, verbose: bool = False
) -> Dict:
    """Calculate the answer rate of a specific run in the experiment series."""

    df = convert_run_results_to_dataframe(
        series, run_index=run_index, database_name=database_name
    )
    TP = df["TP"].sum()
    FP = df["FP"].sum()
    TN = df["TN"].sum()
    FN = df["FN"].sum()

    FP_ids = df[df["FP"] == 1]["id"].tolist()
    FN_ids = df[df["FN"] == 1]["id"].tolist()
    undecisive_ids = df[df["Undecisive"] == 1]["id"].tolist()

    total_answers = len(df)
    decisive_answers = total_answers - df["Undecisive"].sum()
    correct_answers = df["TP"].sum() + df["TN"].sum()

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    if verbose:
        st.write(f"Total answers: {total_answers}")
        st.write(f"Decisive answers: {decisive_answers}")
        st.write(f"Correct answers: {correct_answers}")
        st.write(
            f"Answer rate: {decisive_answers / total_answers if total_answers > 0 else 0.0:.2%}"
        )
        st.write(f"Recall: {recall:.2%}")
        st.write(f"Precision: {precision:.2%}")
        st.write(f"Accuracy: {accuracy:.2%}")
        st.write(f"FP IDs: {FP_ids}")
        st.write(f"FN IDs: {FN_ids}")
        st.write(f"Undecisive IDs: {undecisive_ids}")

    metrics = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "FP_ids": FP_ids,
        "FN_ids": FN_ids,
        "undecisive_ids": undecisive_ids,
        "total_answers": total_answers,
        "decisive_answers": decisive_answers,
        "correct_answers": correct_answers,
        "answer_rate": decisive_answers / total_answers if total_answers > 0 else 0.0,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }
    return metrics

    # return correct_answers / total_answers if total_answers > 0 else 0.0


# def calculate_metrics_per_experiment(
#     series: ExperimentSeries, database_name: str
# ) -> Dict:
#     """Calculate metrics for each experiment in the series."""
#     for experiment_id in series.experiment_ids:
#         experiment = handler.storage.get_experiment(experiment_id)
#         if not experiment.status == Status.COMPLETED:
#             continue
#         for run_id in experiment.run_ids:
#             run = handler.storage.get_run(run_id)
#             if not run.status == Status.COMPLETED:
#                 continue
#             metrics_run = calculate_metrics_per_run(
#                 series, run_index=0, database_name=database_name
#             )
#             st.write(f"Metrics for Experiment {experiment_id}, Run {run_id}:")
#             st.write(metrics)


def calculate_metrics_per_series(series: ExperimentSeries, database_name: str) -> Dict:
    run_metrics: List[Dict] = []

    # get number of runs
    number_of_runs = len(
        handler.storage.get_experiment(series.experiment_ids[0]).run_ids
    )

    for i in range(number_of_runs):
        run_metrics.append(
            calculate_metrics_per_run(series, run_index=i, database_name=database_name)
        )

    # calculate averages and standard deviations
    averages = {
        "TP": np.mean([run["TP"] for run in run_metrics]),
        "FP": np.mean([run["FP"] for run in run_metrics]),
        "TN": np.mean([run["TN"] for run in run_metrics]),
        "FN": np.mean([run["FN"] for run in run_metrics]),
        "total_answers": np.mean([run["total_answers"] for run in run_metrics]),
        "decisive_answers": np.mean([run["decisive_answers"] for run in run_metrics]),
        "correct_answers": np.mean([run["correct_answers"] for run in run_metrics]),
        "answer_rate": np.mean([run["answer_rate"] for run in run_metrics]),
        "recall": np.mean([run["recall"] for run in run_metrics]),
        "precision": np.mean([run["precision"] for run in run_metrics]),
        "accuracy": np.mean([run["accuracy"] for run in run_metrics]),
    }

    std_devs = {
        "TP": np.std([run["TP"] for run in run_metrics]),
        "FP": np.std([run["FP"] for run in run_metrics]),
        "TN": np.std([run["TN"] for run in run_metrics]),
        "FN": np.std([run["FN"] for run in run_metrics]),
        "total_answers": np.std([run["total_answers"] for run in run_metrics]),
        "decisive_answers": np.std([run["decisive_answers"] for run in run_metrics]),
        "correct_answers": np.std([run["correct_answers"] for run in run_metrics]),
        "answer_rate": np.std([run["answer_rate"] for run in run_metrics]),
        "recall": np.std([run["recall"] for run in run_metrics]),
        "precision": np.std([run["precision"] for run in run_metrics]),
        "accuracy": np.std([run["accuracy"] for run in run_metrics]),
    }

    st.write(
        f"This section provides an overview on the combined metrics for {number_of_runs} runs in the selected series."
    )

    st.write(
        f"Total answers across all runs: {averages['total_answers']} +- {std_devs['total_answers']:.2f}"
    )
    st.write(
        f"Decisive answers across all runs: {averages['decisive_answers']} +- {std_devs['decisive_answers']:.2f}"
    )
    st.write(
        f"Correct answers across all runs: {averages['correct_answers']} +- {std_devs['correct_answers']:.2f}"
    )

    st.write(
        f"Answer rate across all runs: {averages['answer_rate']:.2%} +- {std_devs['answer_rate']:.2%}"
    )
    st.write(f"TP across all runs: {averages['TP']} +- {std_devs['TP']:.2f}")
    st.write(f"FP across all runs: {averages['FP']} +- {std_devs['FP']:.2f}")
    st.write(f"TN across all runs: {averages['TN']} +- {std_devs['TN']:.2f}")
    st.write(f"FN across all runs: {averages['FN']} +- {std_devs['FN']:.2f}")
    st.write(
        f"Recall across all runs: {averages['recall']:.2%} +- {std_devs['recall']:.2%}"
    )
    st.write(
        f"Precision across all runs: {averages['precision']:.2%} +- {std_devs['precision']:.2%}"
    )
    st.write(
        f"Accuracy across all runs: {averages['accuracy']:.2%} +- {std_devs['accuracy']:.2%}"
    )

    return {
        "averages": averages,
        "std_devs": std_devs,
        "run_metrics": run_metrics,
    }


# def calculate_run_accuracy(series: ExperimentSeries) -> float:
#     """Calculate the accuracy of runs in the experiment series."""
#     total_runs = 0
#     correct_runs = 0

#     for experiment_id in series.experiment_ids:
#         experiment = handler.storage.get_experiment(experiment_id)
#         for run_id in experiment.run_ids:
#             run = handler.storage.get_run(run_id)
#             total_runs += 1
#             if run.success:


def format_triple(triple: Optional[Dict]) -> str:
    """Format a triple dictionary into a string."""
    if triple is not None:
        return f"({triple['head']['value']}, {triple['edge']['value']}, {triple['tail']['value']})"
    return None


# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Result Visualization")
    st.sidebar.header("Selection")

    # Create a dropdown to select the database
    database_options = handler.storage.client._client.list_database_names()
    selected_db = st.sidebar.selectbox("Select Database", database_options)
    handler.storage.set_database(selected_db)

    available_series = handler.storage.client.get_all_entries_in_collection(
        "experiment_series", selected_db
    )
    available_series_ids = [entry["_id"] for entry in available_series]

    # Create a dropdown to select the experiment series
    selected_series = st.sidebar.selectbox(
        "Select Experiment Series", available_series_ids
    )

    if not selected_series:
        st.error("No experiment series found in the selected database.")
        return

    series = handler.storage.get_series(selected_series)

    # Experiment Runs Selection
    num_runs = len(handler.storage.get_experiment(series.experiment_ids[0]).run_ids)
    selected_run = st.sidebar.selectbox(
        "Select Run", ["Overview"] + list(range(num_runs))
    )

    if selected_run == "Overview":
        st.subheader(f"Overview of Experiment Series {selected_series}")
        calculate_metrics_per_series(series, database_name=selected_db)
        return

    entries = handler.storage.client.get_all_entries_in_collection(
        "dataset", selected_db
    )

    if not entries:
        st.error("No dataset entries found in the selected database.")
        return

    ids = [(entry["id"], entry["_id"]) for entry in entries]

    # Create a dropdown to select the dataset ID
    dataset_ids = ["Overview"] + [entry[0] for entry in ids]
    selected_id = st.sidebar.selectbox("Select Dataset ID", dataset_ids)

    # Filter the dataset and results for the selected ID
    dataset_entry = next(
        (entry for entry in entries if entry["id"] == selected_id), None
    )
    # result_entry = results[results["id"] == selected_id]

    if selected_id == "Overview":
        st.subheader(f"Overview of Experiment Series {selected_series}")
        st.write("This section provides an overview of all dataset entries.")
        calculate_metrics_per_run(
            series, run_index=selected_run, database_name=selected_db, verbose=True
        )

    else:
        if dataset_entry:
            # Display the query
            st.subheader(f"Dataset entry: {selected_id}")
            st.write(f"**Query:** {dataset_entry['query']}")
            st.write(f"**Answer:** {dataset_entry['answer']}")
            st.write(f"**Inference Rule:** {dataset_entry['Inference Rule']}")
            reasoning_strategies = dataset_entry.get("Reasoning Strategy", [])
            for i, strategy in enumerate(reasoning_strategies, 1):
                st.write(f"**Reasoning Strategy {i}:** {strategy}")
            kg_triples = dataset_entry.get("KG Triples", [])
            for i, triple in enumerate(kg_triples, 1):
                st.write(f"**KG Triple {i}:** {triple}")
            reasoning_steps = dataset_entry.get("Reasoning Steps", [])
            if reasoning_steps:
                st.markdown("#### Reasoning Steps")
                for i, step in enumerate(reasoning_steps, 1):
                    st.write(f"**Step {i}:** {step}")
            else:
                st.write("No reasoning steps available in the dataset.")

            results = get_run_results(selected_id, series)
            st.subheader("Agent Results:")

            for i, result in enumerate(results):
                st.markdown(f"#### Run {i}:")
                answer = result.get("answer")
                st.write(f"**Answer:** {answer['answer']}")
                st.write(f"**Justification:** {answer['justification']}")
                st.write(f"**Iteration:** {result['final_state']['iteration']}")
                reasoning_steps = result["final_state"].get("reasoningSteps", [])
                Valid = []
                anchors = []
                attempts = []
                triples = []
                implications = []
                for j, step in enumerate(reasoning_steps, 1):
                    attempts_dict = {}
                    anchors.append(step["anchor"]["anchor"]["value"])
                    attempts_dict["anchor"] = step["anchor"]["attempt"]
                    if step.get("relation_selected"):
                        attempts_dict["relation"] = step["relation_selected"]["attempt"]
                    result = step.get("result")
                    if result:
                        attempts_dict["result"] = result["attempt"]
                        if result.get("triples"):
                            triples.append(
                                [format_triple(triple) for triple in result["triples"]]
                            )
                        else:
                            triples.append(["None, None, None"])
                        if result["result"].get("implications"):
                            implications.append(result["result"]["implications"])
                        else:
                            implications.append([""])
                    else:
                        triples.append(["None, None, None"])
                        implications.append([""])
                    attempts.append(attempts_dict)

                data = {
                    "Anchors": anchors,
                    "Triples": triples,
                    "Implications": implications,
                    "Attempts": attempts,
                }

                df = pd.DataFrame(data)
                st.table(df)

            # st.write(f"**Selected Relation:** {step['relations_selected']['relations'][0]["relation"]}")
            # st.write(f"**Result:** {result['output']}")
            # st.write(f"**Explanation:** {result['explanation']}")

        # st.write(f"**Reasoning Strategy:** {dataset_entry['answer']}")

    #     # Display the final answers
    #     st.subheader("Final Answer Comparison")
    #     st.write(f"**Dataset Answer:** {dataset_entry['answer']}")
    #     st.write(f"**Agent Answer:** {result_entry.iloc[0]['result']}")

    #     # Explanation comparison
    #     st.subheader("Explanation Comparison")
    #     st.write(f"**Dataset Explanation:** {dataset_entry['Inference Rule']}")
    #     st.write(f"**Agent Explanation:** {result_entry.iloc[0]['explanation']}")

    #     # Display the reasoning steps from the dataset
    #     st.subheader("Reasoning Steps (Dataset)")
    #     dataset_reasoning_steps = dataset_entry.get("Reasoning Steps", [])
    #     if dataset_reasoning_steps:
    #         for i, step in enumerate(dataset_reasoning_steps, 1):
    #             st.markdown(f"**Step {i}:** {step}")
    #     else:
    #         st.write("No reasoning steps available in the dataset.")

    #     # Display the reasoning steps from the agent
    #     st.subheader("Reasoning Steps (Agent)")
    #     agent_reasoning_steps = result_entry.iloc[0]["reasoning_steps"]
    #     if agent_reasoning_steps:
    #         # Parse reasoning steps if stored as a string

    #         parsed_data = ast.literal_eval(agent_reasoning_steps)
    #         # agent_reasoning_steps = json.loads(agent_reasoning_steps)
    #         # print(parsed_data)
    #         for i, step in enumerate(parsed_data, 1):
    #             # print(i, step)
    #             st.markdown(f"**Step {i}:** {step['implications']}")
    #     else:
    #         st.write("No reasoning steps available from the agent.")

    # else:
    #     st.write("No matching data found for the selected ID.")


# Run the app
if __name__ == "__main__":
    main()
