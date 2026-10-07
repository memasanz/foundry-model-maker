# Option 2 — GitHub Copilot ML-trainer agent, hosted on Microsoft Foundry

Same use case as [Option 1](../prompt-agent) (train a model from a dataset and export
it) — but here the agent is a **GitHub Copilot** agent **hosted on Foundry**, built with
the Microsoft **Agent Framework** (`agent-framework-github-copilot`) and served over the
**Responses protocol**.

The key difference from Option 1: the GitHub Copilot CLI **harness runs inside the hosted
container's micro-VM**, so the agent writes and executes real Python to train with **any**
library baked into the image (scikit-learn, XGBoost, LightGBM, pandas, MLflow, …). Because
Foundry gives each session its own micro-VM, there's no file build-up or cross-user
clobbering to manage — the VM is torn down when the session ends.

Inference is routed to **your own Foundry / Azure OpenAI deployment** via **BYOK** (Bring
Your Own Key) + **Microsoft Entra OAuth** — no API keys, no per-user Copilot subscription.

```
 client ──▶ Foundry Agent Service ──▶ container (this agent, its own micro-VM)
                                          │  GitHubCopilotAgent (agent_framework.github)
                                          │    └─ spawns the `copilot` CLI harness
                                          │         └─ writes + runs Python to TRAIN
                                          │         └─ BYOK provider → your Foundry endpoint
                                          │              └─ bearer_token_provider() ← Entra OAuth
                                          ▼
                              Foundry / Azure OpenAI  (gpt-5.x deployment)
```

## Layout

```
copilot-agent/                       ← project root: run azd commands here
├─ azure.yaml                        ← hosted service + Foundry ai-project config
├─ README.md
├─ data/                             ← sample datasets for testing
└─ src/
   └─ copilot-ml-trainer/            ← the agent container
      ├─ main.py                     ← GitHubCopilotAgent + ML instructions + BYOK + Entra OAuth
      ├─ requirements.txt            ← af-github/hosting + ML stack (sklearn, xgboost, lightgbm, mlflow, …)
      ├─ uv.toml                     ← pins github-copilot-sdk==1.0.11 (BYOK OAuth fix)
      ├─ Dockerfile                  ← Python + Node + @github/copilot CLI + ML stack
      └─ .env.example
```

## What the agent does

Given a tabular dataset (pasted inline, or a path to a file in its working directory), the
agent trains a model and exports three artifacts into its working directory:

- `model.pkl` — plain joblib pickle of the fitted model/pipeline
- `mlflow_model/` — MLflow model dir (serialized model **+** `conda.yaml` / `requirements.txt`
  / `python_env.yaml`, so the artifact carries its own runtime environment)
- `metrics.json` — task type, library/model, target column, evaluation metrics

> **Scope note (skeleton-first):** this version gets the hosted agent deployable and able
> to train in its micro-VM. Streaming a dataset file **in** and the exported `model.pkl`
> **out** over the Responses protocol is the next iteration — see
> [File upload / export](#file-upload--export-next-step).

## Configuration

Set via `azd env set` (sourced into `azure.yaml` → container env). `FOUNDRY_*`/`AGENT_*`
names are reserved by the platform, so this agent uses `BYOK_*` names.

| Variable | Required | Description |
| --- | --- | --- |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | ✅ | Model deployment, e.g. `gpt-5.5` |
| `BYOK_MODEL_ENDPOINT` | ✅ | OpenAI v1 inference URL, e.g. `https://<resource>.openai.azure.com/openai/v1/` |
| `BYOK_TOKEN_SCOPE` | ⚠️ | Entra scope — **must match the endpoint host** (table below) |
| `BYOK_PROVIDER_TYPE` | optional | `openai` (default) or `azure` |
| `BYOK_WIRE_API` | optional | `responses` (default, gpt-5.x) or `completions` |
| `COPILOT_ALLOWED_PERMISSIONS` | optional | Comma-separated harness permission **kinds** to approve; any other kind is denied. Kinds: `shell` (run code), `read`, `write`, `url` (fetch web), `mcp`. Default `shell,read,write` (what training needs, no web/MCP). Add `url` to let the agent pull datasets from the internet. |

| Endpoint host | `BYOK_TOKEN_SCOPE` |
| --- | --- |
| `*.openai.azure.com` | `https://cognitiveservices.azure.com/.default` |
| `*.services.ai.azure.com` | `https://ai.azure.com/.default` |

## Deploy

```powershell
cd copilot-agent
azd auth login

# Target project + model + BYOK (this repo reuses the admin-5729 Foundry project)
azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME "gpt-5.5"
azd env set BYOK_MODEL_ENDPOINT            "https://admin-5729-resource.openai.azure.com/openai/v1/"
azd env set BYOK_TOKEN_SCOPE              "https://cognitiveservices.azure.com/.default"
azd env set COPILOT_ALLOWED_PERMISSIONS   "shell,read,write"   # add "url" to allow web fetches

# If bringing your own ACR (admin-5729 already has one):
azd env set AZURE_CONTAINER_REGISTRY_ENDPOINT    "<name>.azurecr.io"
azd env set AZURE_CONTAINER_REGISTRY_RESOURCE_ID "<ACR_RESOURCE_ID>"

azd deploy
```

### RBAC (three identities)

1. **You** (running `azd`): *Foundry Project Manager* on the project, *Container Registry
   Tasks Contributor* on the ACR, *Cognitive Services OpenAI User* on the account.
2. **Foundry project** system identity: *AcrPull* on the ACR (so Foundry can pull the image).
3. **Per-agent instance identity** (discoverable only **after** the first `azd deploy`):
   *Cognitive Services OpenAI User* on the account — else inference returns
   `HTTP 401 … Authentication failed with provider`.

```bash
# After first deploy: grant the per-agent identity model access
AGENT_MI=$(az rest --method get \
  --url "<PROJECT_ENDPOINT>/agents/copilot-ml-trainer/versions/<version>?api-version=v1" \
  --query instance_identity.principal_id -o tsv)
az role assignment create --assignee-object-id "$AGENT_MI" \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" --scope "<FOUNDRY_ACCOUNT_RESOURCE_ID>"
```

## Invoke

```bash
azd ai agent invoke "Train a classifier on the iris dataset and report accuracy."
azd ai agent monitor --type console --follow   # watch the harness train, live
```

## Dependency note: `github-copilot-sdk` override

`agent-framework-github-copilot==1.0.2` pins `github-copilot-sdk==1.0.2`, whose
`ProviderConfig` **lacks `bearer_token_provider`** — the OAuth callback is silently dropped
and every BYOK call 401s. We override to `1.0.11` in **two** places: `uv.toml` (local
`azd ai agent run`) and the `Dockerfile` (deployed image). Remove both once af-github
depends on `github-copilot-sdk>=1.0.11`.

## File upload / export (next step)

The current skeleton trains inside the micro-VM and reports where the files are. The next
iteration wires the Responses protocol so a user can upload a dataset file and download the
exported `model.pkl` / `mlflow_model.zip` — mirroring what Option 1 does with the Code
Interpreter container. Until then, test with small datasets pasted inline.

## References

- [Deploy a hosted agent](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/deploy-hosted-agent)
- [Copilot SDK — BYOK auth](https://github.com/github/copilot-sdk/blob/main/docs/auth/byok.md)
- [Agent Framework](https://github.com/microsoft/agent-framework)
