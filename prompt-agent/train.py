"""Train a model with the Foundry ML-trainer agent from the command line.

Give it a CSV or Excel file, it uploads the file to the agent, the agent trains a
scikit-learn model in the Code Interpreter sandbox, and this script downloads the
resulting model.pkl next to your dataset.

    python train.py data/housing.csv
    python train.py data/housing.csv median_house_value
    python train.py data/iris.xlsx species classification

The target column and task type are optional. If omitted the agent infers the
task (regression / classification / clustering) and, for supervised tasks, the
target column.
"""

import os
import sys

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["MODEL_DEPLOYMENT_NAME"]
AGENT_NAME = os.environ["AGENT_NAME"]

# Dataset formats the Code Interpreter agent can load (pandas read_csv / read_excel).
ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def find_container_ids(response) -> list[str]:
    """Collect all distinct Code Interpreter container ids referenced in the output."""
    ids: list[str] = []
    for item in response.output or []:
        container_id = getattr(item, "container_id", None)
        if container_id and container_id not in ids:
            ids.append(container_id)
    return ids


def download_named_file(
    openai_client, container_ids: list[str], filename: str, out_path: str
) -> bool:
    """Download a file by exact basename, searching every container the run used.

    Matches on the basename so that, e.g., the top-level ``model.pkl`` is picked rather
    than a copy nested inside ``mlflow_model/``. If several match, the shortest path wins.
    """
    for container_id in container_ids:
        files = openai_client.containers.files.list(container_id=container_id)
        matches = [
            f
            for f in files.data
            if os.path.basename((getattr(f, "path", "") or "")) == filename
        ]
        if not matches:
            continue
        chosen = min(matches, key=lambda f: len(getattr(f, "path", "") or ""))
        content = openai_client.containers.files.content.retrieve(
            file_id=chosen.id, container_id=container_id
        )
        content.write_to_file(out_path)
        return True
    return False


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python train.py <data_path> [target_column] [task_type]")
        print("       data_path: a .csv, .xlsx or .xls file")
        sys.exit(1)

    data_path = sys.argv[1]
    target_column = sys.argv[2] if len(sys.argv) > 2 else None
    task_type = sys.argv[3] if len(sys.argv) > 3 else None
    if not os.path.isfile(data_path):
        print(f"file not found: {data_path}")
        sys.exit(1)
    if os.path.splitext(data_path)[1].lower() not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        print(f"unsupported file type '{os.path.splitext(data_path)[1]}'. Use one of: {allowed}")
        sys.exit(1)

    client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    openai_client = client.get_openai_client()

    print(f"Uploading {data_path} ...")
    with open(data_path, "rb") as fh:
        uploaded = openai_client.files.create(
            file=(os.path.basename(data_path), fh), purpose="assistants"
        )

    prompt = "Train a model on the uploaded dataset and save it as model.pkl."
    if target_column:
        prompt += f" The target column is '{target_column}'."
    if task_type:
        prompt += f" Treat this as a {task_type} task."

    print("Training (this runs in the Code Interpreter sandbox) ...")
    response = openai_client.responses.create(
        extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"}},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": uploaded.id},
                ],
            }
        ],
    )

    print("\n--- Agent summary ---")
    print(response.output_text)
    print("---------------------\n")

    container_ids = find_container_ids(response)
    if not container_ids:
        print("No Code Interpreter container was created; nothing to download.")
        return

    out_dir = os.path.dirname(os.path.abspath(data_path))

    pkl_path = os.path.join(out_dir, "model.pkl")
    if download_named_file(openai_client, container_ids, "model.pkl", pkl_path):
        print(f"Saved trained model -> {pkl_path}")
        print("Load it with:  import joblib; model = joblib.load('model.pkl')")
    else:
        print("The agent did not produce a model.pkl file in this run.")

    mlflow_path = os.path.join(out_dir, "mlflow_model.zip")
    if download_named_file(openai_client, container_ids, "mlflow_model.zip", mlflow_path):
        print(f"Saved MLflow model  -> {mlflow_path}")
        print("Unzip it to get MLmodel + conda.yaml / requirements.txt / python_env.yaml.")
    else:
        print("The agent did not produce an mlflow_model.zip in this run.")


if __name__ == "__main__":
    main()
