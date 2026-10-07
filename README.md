# Found Agents - Prompt & Hosted - Building models with your data...

A collection of **agent-based model builders** on Microsoft Foundry. Each
subdirectory is a self-contained option for "give the agent data, get a trained
model back."

## Options

| Directory | Approach |
|-----------|----------|
| [`prompt-agent/`](prompt-agent/) | Foundry **prompt agent** + Code Interpreter tool. Upload a CSV, it trains a scikit-learn model (regression / classification / clustering) and returns `model.pkl`. Code runs in Foundry's cloud sandbox. Works in the Foundry playground web chat or via CLI. |
| [`copilot-agent/`](copilot-agent/) | **GitHub Copilot agent hosted on Foundry** (Agent Framework + Responses protocol). The Copilot CLI harness runs inside the hosted container's per-session micro-VM, so it can use any ML library (scikit-learn, XGBoost, LightGBM, ...) and logs the model with **MLflow** — the export ships with its own environment files plus `model.pkl`. Inference routes to your Foundry deployment via BYOK + Entra OAuth. |
| [`copilot-toolbox-agent/`](copilot-toolbox-agent/) | **GitHub Copilot agent hosted on Foundry that runs code in a Foundry Toolbox Code Interpreter**. Same Copilot harness + BYOK inference as `copilot-agent/`, but instead of executing training code in its own container it calls the toolbox's `code_interpreter` MCP tool, so code runs in Foundry's managed sandbox (a full Python env it can `pip install` into) — not limited by the container, and isolated from the host. |

More options can be added as sibling directories (e.g. a different ML framework,
a local harness runner, etc.).

## Conventions

- Each option directory has its own `README.md`, `requirements.txt`, and scripts.
- Secrets live in a per-directory `.env` (git-ignored).
- Trained `model.pkl` artifacts are git-ignored.

> Built with the microsoft-foundry skill. See `AGENTS.md`.
