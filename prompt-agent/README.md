# ml-trainer — Foundry Code Interpreter agent

A Microsoft Foundry **prompt agent** with the **Code Interpreter** tool. Give it a
dataset (CSV), it trains an appropriate scikit-learn model — **regression,
classification, or clustering** (auto-detected, or tell it which) — in a sandbox
and returns a downloadable `model.pkl`.

> This project was built with the microsoft-foundry skill. Before working on or
> answering questions about Foundry agents, read the microsoft-foundry skill first.

## What's here

| File | Purpose |
|------|---------|
| `create_agent.py` | Creates/updates the `ml-trainer` agent (run once, or after editing its instructions). |
| `train.py` | CLI: upload a CSV -> agent trains -> downloads `model.pkl` next to the CSV. |
| `predict.py` | CLI: score new rows with a downloaded `model.pkl`. |
| `.env` | Project endpoint, model deployment, agent name. |
| `requirements.txt` | Deps to run the scripts here. |
| `requirements-loader.txt` | Pinned deps (scikit-learn 1.1.3) to LOAD a downloaded model. |
| `data/housing.csv` | Small sample dataset. |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --index-url https://packagefeedproxy.microsoft.io/pypi/simple
az login   # must be logged into the subscription that owns the Foundry project
```

## Use it

**Option A — Foundry portal (web chat).** Open the project in the Foundry portal
-> Agents -> `ml-trainer` -> Playground. Attach a `.csv`, say "train a regressor,
target column is price, return model.pkl", then download the `model.pkl` link.
CSV is a supported Code Interpreter upload type.

**Option B — command line.**

```powershell
python create_agent.py                      # create/update the agent
python train.py data\housing.csv price      # regression -> writes data\model.pkl
python train.py data\iris.csv species classification
```

## Loading the downloaded model (important)

The Code Interpreter sandbox trains with **scikit-learn 1.1.3**, and scikit-learn
pickles are version-sensitive. Load `model.pkl` in a matching environment or
unpickling may fail with an "incompatible dtype" error:

```powershell
python -m venv .venv-loader           # Python 3.8-3.11 (sklearn 1.1.3 requirement)
.\.venv-loader\Scripts\Activate.ps1
pip install -r requirements-loader.txt --index-url https://packagefeedproxy.microsoft.io/pypi/simple
python predict.py data\model.pkl data\housing.csv
```

The agent reports the exact library versions it used at the end of each run.
