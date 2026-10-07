# Copyright (c) Microsoft. All rights reserved.
"""Option 2: a GitHub Copilot ML-trainer agent, hosted on Microsoft Foundry.

This is the same use case as Option 1 (../../prompt-agent): train a model from a
dataset and export it. The difference is *where the code runs*. Here the GitHub
Copilot CLI harness runs INSIDE this Foundry-hosted container's micro-VM, so the
agent can write and execute real Python to train with any library already baked
into the image (scikit-learn, XGBoost, LightGBM, pandas, MLflow, ...).

Inference is routed to your own Foundry / Azure OpenAI deployment via BYOK (Bring
Your Own Key) + Microsoft Entra OAuth -- no API keys, no per-user Copilot subscription.
The agent is served over the Responses protocol by ResponsesHostServer, so it plugs
into the Foundry playground and `azd ai agent invoke`.
"""

import os
from typing import Any, Literal, cast

from agent_framework.github import GitHubCopilotAgent, GitHubCopilotOptions
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity.aio import DefaultAzureCredential
from copilot.generated.rpc import PermissionDecisionReject
from copilot.session import (
    PermissionHandler,
    PermissionInvocation,
    PermissionRequestResult,
    ProviderConfig,
)
from copilot.session_events import PermissionRequest
from dotenv import load_dotenv

# Load environment variables from .env file (local runs only; no-op in the container).
load_dotenv()


INSTRUCTIONS = """\
You are an ML engineer running inside a container with a full Python environment.
Your job: given a tabular dataset, train a model and export it.

Environment already available (no install needed for these):
  python3 with scikit-learn, pandas, numpy, joblib, mlflow, xgboost, lightgbm and
  openpyxl. Run code with the `python3` on PATH. You may install extra packages with
  `python3 -m pip install <pkg>` if you genuinely need them.

Working with data:
- If the user pastes a dataset inline (CSV text or a small table), write it to a file
  in the working directory first, then load it.
- If the user gives a path to a file already in the working directory, use it.
- Load CSV with pandas.read_csv and Excel (.xlsx / .xls) with pandas.read_excel.
- If the target column or task type isn't stated, infer the most sensible one and say
  which you chose.

Training:
- Pick the best library/model for the task (scikit-learn, XGBoost, LightGBM,
  statsmodels, ...). Do sensible preprocessing (encode categoricals, impute/drop
  missing, scale when helpful) and a proper train/test split. Avoid data leakage.

Export these three artifacts into the working directory:
1. model.pkl  -- a plain joblib pickle of the fitted model (or the FULL pipeline, so it
   can score raw rows end to end):  import joblib; joblib.dump(model, "model.pkl")
2. mlflow_model/ -- an MLflow model directory that records the exact runtime environment
   (conda.yaml / requirements.txt / python_env.yaml). Use the matching flavor, e.g.
   mlflow.sklearn.save_model(model, path="mlflow_model")  (or mlflow.xgboost /
   mlflow.lightgbm / mlflow.pyfunc).
3. metrics.json -- task type, chosen library+model, the target column, and evaluation
   metrics (R2/RMSE for regression, accuracy/F1 for classification, silhouette for
   clustering).

When finished, print a concise summary: task type, library + model, key metrics, the
target column, and confirm that model.pkl, mlflow_model/ and metrics.json exist in the
working directory. Tell the user where the files are so they can be retrieved.
"""


# The Copilot CLI requests permission per capability "kind": shell, read, write, mcp, url.
# Rather than a blunt approve-all, the operator chooses which kinds to allow via
# COPILOT_ALLOWED_PERMISSIONS (comma-separated); every other kind is denied. The default
# grants exactly what training needs -- run code (shell) and load a local dataset / write
# the exported model (read, write) -- while withholding web access (url) and MCP (mcp).
# FOUNDRY_* / AGENT_* env names are reserved by the platform, hence the COPILOT_ prefix.
DEFAULT_ALLOWED_PERMISSIONS = "shell,read,write"
KNOWN_PERMISSION_KINDS = {"shell", "read", "write", "mcp", "url"}


def build_permission_handler():
    """Build a handler that approves the permission kinds named in COPILOT_ALLOWED_PERMISSIONS
    and denies the rest. Each decision is logged (`azd ai agent monitor --type console`)."""
    raw = os.getenv("COPILOT_ALLOWED_PERMISSIONS") or DEFAULT_ALLOWED_PERMISSIONS
    allowed = {k.strip().lower() for k in raw.split(",") if k.strip()}
    unknown = allowed - KNOWN_PERMISSION_KINDS
    if unknown:
        raise RuntimeError(
            f"Unknown permission kind(s) in COPILOT_ALLOWED_PERMISSIONS: {sorted(unknown)}. "
            f"Valid kinds: {sorted(KNOWN_PERMISSION_KINDS)}."
        )
    print(f"Allowed permission kinds: {sorted(allowed)}", flush=True)

    def handler(
        request: PermissionRequest, context: PermissionInvocation
    ) -> PermissionRequestResult:
        kind = getattr(request, "kind", "")
        if kind in allowed:
            print(f"  [permission: {kind}] -> approved", flush=True)
            return PermissionHandler.approve_all(request, context)
        print(f"  [permission: {kind}] -> denied", flush=True)
        return PermissionDecisionReject(
            feedback=f"Permission '{kind}' is not allowed by COPILOT_ALLOWED_PERMISSIONS."
        )

    return handler


def main():
    # Model deployment name. BYOK also requires the model to be set at the session level.
    model_name = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME") or os.getenv("FOUNDRY_MODEL_NAME")
    if not model_name:
        raise RuntimeError(
            "Model deployment name is not configured. Set "
            "AZURE_AI_MODEL_DEPLOYMENT_NAME or FOUNDRY_MODEL_NAME."
        )

    # BYOK (Bring Your Own Key): route the GitHub Copilot agent's inference through your own
    # Microsoft Foundry / Azure OpenAI endpoint instead of the default GitHub Copilot backend.
    # Inference is served and billed by Foundry, so no per-user GitHub Copilot subscription is
    # required. See https://github.com/github/copilot-sdk/blob/main/docs/auth/byok.md
    base_url = os.environ["BYOK_MODEL_ENDPOINT"]  # e.g. https://<resource>.services.ai.azure.com/openai/v1/
    provider_type = cast(
        Literal["openai", "azure", "anthropic"],
        os.getenv("BYOK_PROVIDER_TYPE", "openai"),
    )
    # Use "responses" for newer models (e.g. gpt-5.x); "completions" for older models.
    wire_api = cast(Literal["responses", "completions"], os.getenv("BYOK_WIRE_API", "responses"))
    # Microsoft Entra scope for Foundry model inference.
    token_scope = os.getenv("BYOK_TOKEN_SCOPE", "https://ai.azure.com/.default")

    # OAuth (Microsoft Entra ID) instead of an API key. The Copilot SDK calls this callback
    # before each outbound request; DefaultAzureCredential caches and refreshes tokens
    # automatically, so the hosted agent keeps working past the ~1h token lifetime. In Foundry
    # this uses the container's managed identity; locally it falls back to your `az login`.
    credential = DefaultAzureCredential()

    async def get_bearer_token(_args: Any) -> str:
        token = await credential.get_token(token_scope)
        return token.token

    provider: ProviderConfig = {
        "type": provider_type,
        "base_url": base_url,
        "bearer_token_provider": get_bearer_token,
        "wire_api": wire_api,
        "model_id": model_name,
    }

    agent = GitHubCopilotAgent(
        instructions=INSTRUCTIONS,
        default_options=GitHubCopilotOptions(
            model=model_name,
            provider=provider,
            on_permission_request=build_permission_handler(),
            timeout=1800,
        ),
    )

    # GitHubCopilotAgent is a custom SupportsAgentRun (not a RawAgent), so hosting
    # cannot enforce server-side history storage. Use history_source="agent" to let
    # the agent manage its own conversation history. Without this, newer
    # agent-framework-foundry-hosting versions raise a RuntimeError at startup and the
    # container fails its readiness check (HTTP 424 session_not_ready on invoke).
    server = ResponsesHostServer(agent, history_source="agent")
    server.run()


if __name__ == "__main__":
    main()
