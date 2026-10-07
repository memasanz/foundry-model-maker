# Option 3 — GitHub Copilot agent that executes code via a Foundry Toolbox

Same use case as [Option 1](../prompt-agent) and [Option 2](../copilot-agent) (train a
model from a dataset and export it) — but here the agent is a **GitHub Copilot** agent
**hosted on Foundry** that runs **all code inside a Foundry Toolbox Code Interpreter
sandbox** instead of a local shell.

- **Option 2** runs the Copilot CLI harness *and* executes training code **inside its own
  container** (micro-VM), limited to the libraries baked into that image.
- **Option 3** keeps the Copilot harness, but gives it the Foundry **Toolbox** MCP endpoint
  as a tool. The agent calls the toolbox's `code_interpreter` to run Python in Foundry's
  **managed, isolated sandbox** — a full Python environment it can `pip install` into at
  will, so it's **not limited** by this container, and nothing it runs touches the host.

Inference is routed to **your own Foundry / Azure OpenAI deployment** via **BYOK** (Bring
Your Own Key) + **Microsoft Entra OAuth** — no API keys, no per-user Copilot subscription.

```
 client ──▶ Foundry Agent Service ──▶ container (this agent)
                                          │  GitHubCopilotAgent (agent_framework.github)
                                          │    └─ spawns the `copilot` CLI harness
                                          │         ├─ BYOK provider → your Foundry endpoint
                                          │         │     └─ bearer_token_provider() ← Entra OAuth
                                          │         └─ mcp_servers["foundry-toolbox"] (HTTP MCP)
                                          │              │   Authorization: Bearer ⟨Entra token⟩
                                          ▼              ▼
                     Foundry / Azure OpenAI     Foundry Toolbox  →  code_interpreter sandbox
                       (gpt-5.x deployment)                            (runs the training code)
```

## Layout

```
copilot-toolbox-agent/                   ← project root: run azd commands here
├─ azure.yaml                            ← hosted service + Foundry ai-project config
├─ README.md
└─ src/
   └─ copilot-toolbox-trainer/           ← the agent container
      ├─ main.py                         ← GitHubCopilotAgent + toolbox MCP wiring + BYOK + Entra OAuth
      ├─ requirements.txt                ← af-github/hosting + azure-identity (NO local ML stack)
      ├─ uv.toml                         ← pins github-copilot-sdk==1.0.11 (BYOK OAuth fix)
      ├─ Dockerfile                      ← Python + Node + @github/copilot CLI (lightweight)
      └─ .env.example
```

## What the agent does

Given a tabular dataset (pasted inline, or a file referenced by the user), the agent sends
Python to the toolbox `code_interpreter`, which trains a model and produces three artifacts
**in the sandbox**:

- `model.pkl` — plain joblib pickle of the fitted model/pipeline
- `mlflow_model/` — MLflow model dir (serialized model **+** `conda.yaml` / `requirements.txt`
  / `python_env.yaml`, so the artifact carries its own runtime environment)
- `metrics.json` — task type, library/model, target column, evaluation metrics

When the Code Interpreter produces a file and the agent names it, the hosted Responses
adapter emits a native `container_file_citation` annotation (container + file IDs) — the
path a Responses client uses to download the generated file via the container files API.

> **Scope note:** this version is **deployed and drives the toolbox code interpreter end
> to end** (BYOK Entra inference + `code_interpreter` execution — verified: Python 3.11.15 /
> scikit-learn 1.1.3 in the sandbox, model trained and `model.pkl` written). Streaming a
> dataset file **in** and downloading the exported `model.pkl` **out** over the Responses
> protocol (via the `container_file_citation` IDs) is the next iteration.

## Prerequisite: a Toolbox with a Code Interpreter tool

Option 3 needs a Foundry **Toolbox** that exposes a `code_interpreter` tool. This repo uses
one named **`code-exec`** in the `admin-5729` project. Create/verify it via REST (token
scope `https://ai.azure.com/.default`):

```http
POST {PROJECT_ENDPOINT}/toolboxes/code-exec/versions?api-version=v1
Authorization: Bearer ⟨token⟩
Content-Type: application/json

{
  "description": "Code execution sandbox for the GitHub Copilot toolbox agent",
  "tools": [
    { "type": "code_interpreter",
      "description": "Execute Python code in a managed sandbox for data analysis and model training" }
  ]
}
```

The agent's MCP endpoint is then:
`{PROJECT_ENDPOINT}/toolboxes/code-exec/mcp?api-version=v1`

## Configuration

Set via `azd env set` (sourced into `azure.yaml` → container env). `FOUNDRY_*`/`AGENT_*`
names are reserved by the platform, so this agent uses `BYOK_*` / `TOOLBOX_*` names.

| Variable | Required | Description |
| --- | --- | --- |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | ✅ | Model deployment, e.g. `gpt-5.5` |
| `BYOK_MODEL_ENDPOINT` | ✅ | OpenAI v1 inference URL, e.g. `https://<resource>.openai.azure.com/openai/v1/` |
| `BYOK_TOKEN_SCOPE` | ⚠️ | Entra scope — **must match the inference endpoint host** (table below) |
| `TOOLBOX_ENDPOINT` | ✅ | Toolbox MCP URL, e.g. `…/toolboxes/code-exec/mcp?api-version=v1` (or set `TOOLBOX_NAME` + a project endpoint) |
| `TOOLBOX_TOKEN_SCOPE` | optional | Entra scope for the toolbox token (default `https://ai.azure.com/.default`) |
| `COPILOT_ALLOWED_PERMISSIONS` | optional | Harness permission **kinds** to approve; others denied. Kinds: `shell`, `read`, `write`, `url`, `mcp`. Default `mcp,read` keeps code execution in the toolbox (no local `shell`). |

| Inference endpoint host | `BYOK_TOKEN_SCOPE` |
| --- | --- |
| `*.openai.azure.com` | `https://cognitiveservices.azure.com/.default` |
| `*.services.ai.azure.com` | `https://ai.azure.com/.default` |

> **Toolbox token lifetime:** the Copilot harness only accepts a *static* `Authorization`
> header on an MCP server, so the toolbox bearer token is minted **once at startup**. It
> lives ~1h; a very long-lived container may need a restart to re-mint it. A per-request
> refresh would require the Agent-Framework `FoundryToolbox` tool instead of the harness's
> static MCP config — a possible future change.

## Deploy

```powershell
cd copilot-toolbox-agent
azd auth login

azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME "gpt-5.5"
azd env set BYOK_MODEL_ENDPOINT            "https://admin-5729-resource.openai.azure.com/openai/v1/"
azd env set BYOK_TOKEN_SCOPE              "https://cognitiveservices.azure.com/.default"
azd env set TOOLBOX_ENDPOINT             "https://admin-5729-resource.services.ai.azure.com/api/projects/admin-5729/toolboxes/code-exec/mcp?api-version=v1"
azd env set TOOLBOX_TOKEN_SCOPE          "https://ai.azure.com/.default"
azd env set COPILOT_ALLOWED_PERMISSIONS   "mcp,read"

# Reuse the admin-5729 ACR (shared with copilot-agent). AZD_FOUNDRY_ACR_MODE=none stops
# provision from re-creating the ACR's role assignments (otherwise it fails RoleAssignmentExists).
azd env set AZURE_CONTAINER_REGISTRY_ENDPOINT    "<name>.azurecr.io"
azd env set AZURE_CONTAINER_REGISTRY_RESOURCE_ID "<ACR_RESOURCE_ID>"
azd env set AZD_FOUNDRY_ACR_MODE                 "none"

azd provision   # wires FOUNDRY_PROJECT_ENDPOINT for an existing project
# provision blanks AZURE_CONTAINER_REGISTRY_ENDPOINT (ACR mode none) -> re-set it before deploy:
azd env set AZURE_CONTAINER_REGISTRY_ENDPOINT    "<name>.azurecr.io"
azd deploy
```

### RBAC (three identities)

1. **You** (running `azd`): *Foundry Project Manager* on the project, *Container Registry
   Tasks Contributor* on the ACR, *Cognitive Services OpenAI User* on the account.
2. **Foundry project** system identity: *AcrPull* on the ACR (so Foundry can pull the image).
3. **Per-agent instance identity** (discoverable only **after** the first `azd deploy`):
   - *Cognitive Services OpenAI User* on the account — for model inference (else
     `HTTP 401 … Authentication failed with provider`).
   - *Azure AI User* on the project — so it can call the **toolbox** MCP endpoint. **Note:**
     this built-in role is now named **"Foundry User"** (role def id
     `53ca6127-db72-4b80-b1b0-d745d6d5456d`) in current tenants; assign that if "Azure AI
     User" doesn't resolve.

```bash
AGENT_MI=$(az rest --method get \
  --url "<PROJECT_ENDPOINT>/agents/copilot-toolbox-trainer/versions/<version>?api-version=v1" \
  --query instance_identity.principal_id -o tsv)
az role assignment create --assignee-object-id "$AGENT_MI" \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" --scope "<FOUNDRY_ACCOUNT_RESOURCE_ID>"
az role assignment create --assignee-object-id "$AGENT_MI" \
  --assignee-principal-type ServicePrincipal \
  --role "Foundry User" --scope "<FOUNDRY_PROJECT_RESOURCE_ID>"  # formerly "Azure AI User"
```

## Invoke

```bash
azd ai agent invoke "Train a classifier on the iris dataset and report accuracy."
azd ai agent monitor --type console --follow   # watch the toolbox code-interpreter calls, live
```

## Dependency note: `github-copilot-sdk` override

`agent-framework-github-copilot==1.0.2` pins `github-copilot-sdk==1.0.2`, whose
`ProviderConfig` **lacks `bearer_token_provider`** — the OAuth callback is silently dropped
and every BYOK call 401s. We override to `1.0.11` in **two** places: `uv.toml` (local
`azd ai agent run`) and the `Dockerfile` (deployed image). Remove both once af-github
depends on `github-copilot-sdk>=1.0.11`.

## References

- [Microsoft Foundry Toolbox (Agent Framework)](https://learn.microsoft.com/agent-framework/integrations/by-component/tools/foundry-toolbox)
- [Create and manage a toolbox in Foundry](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/toolbox)
- [Code Interpreter tool](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/code-interpreter)
- [GitHub Copilot agent with MCP servers (sample)](https://github.com/microsoft/agent-framework/blob/main/python/samples/02-agents/providers/github_copilot/github_copilot_with_mcp.py)
- [Copilot SDK — BYOK auth](https://github.com/github/copilot-sdk/blob/main/docs/auth/byok.md)
