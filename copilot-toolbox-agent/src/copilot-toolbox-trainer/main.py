# Copyright (c) Microsoft. All rights reserved.
"""Option 3: a GitHub Copilot agent, hosted on Microsoft Foundry, that executes code
through a Foundry Toolbox Code Interpreter tool instead of a local shell.

Same use case as Options 1 and 2 (train a model from a dataset and export it), but the
code runs in Foundry's managed Code Interpreter sandbox -- reached over MCP from the
GitHub Copilot CLI harness -- rather than in this container's micro-VM. The sandbox is a
full, isolated Python environment, so the agent isn't limited to whatever libraries this
image happens to ship, and nothing it runs touches the host container.

How it fits together:
  - Inference: BYOK (Bring Your Own Key) + Microsoft Entra OAuth routes the Copilot
    agent's model calls to your own Foundry / Azure OpenAI deployment (no API key, no
    per-user Copilot subscription).
  - Tools: the Copilot harness is given the Foundry Toolbox's MCP endpoint as an HTTP MCP
    server. The toolbox exposes a `code_interpreter` tool; the agent calls it to run code.
  - Hosting: ResponsesHostServer serves the agent over the Responses protocol, so it plugs
    into the Foundry playground and `azd ai agent invoke`.
"""

import os
from typing import Any, Literal, cast

from agent_framework.github import GitHubCopilotAgent, GitHubCopilotOptions
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential
from copilot.generated.rpc import PermissionDecisionReject
from copilot.session import (
    MCPServerConfig,
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
You are an ML engineer. You do NOT have a local shell. Instead, you run ALL code by
calling the Foundry Toolbox `code_interpreter` tool, which executes Python in a managed,
isolated sandbox. Treat it as your Python runtime: send code, read back stdout/stderr and
any files it produces.

Rules:
- Never try to run shell commands or write code to the local filesystem to execute it.
  Always use the `code_interpreter` tool.
- The sandbox is a full Python environment. Install whatever you need from within your
  code, e.g. `import subprocess; subprocess.run(["pip", "install", "xgboost"])`, then
  import and use it. You are not limited to a fixed set of libraries.

Working with data:
- If the user pastes a dataset inline (CSV text or a small table), recreate it inside the
  sandbox (write it to a file there, or build the DataFrame directly) before loading it.
- If the user references a file, pass its contents into the sandbox.
- Load CSV with pandas.read_csv and Excel (.xlsx / .xls) with pandas.read_excel.
- If the target column or task type isn't stated, infer the most sensible one and say
  which you chose.

Training:
- Pick the best library/model for the task (scikit-learn, XGBoost, LightGBM,
  statsmodels, ...). Do sensible preprocessing (encode categoricals, impute/drop missing,
  scale when helpful) and a proper train/test split. Avoid data leakage.

Export these three artifacts as files inside the sandbox:
1. model.pkl  -- a plain joblib pickle of the fitted model (or the FULL pipeline, so it
   can score raw rows end to end):  import joblib; joblib.dump(model, "model.pkl")
2. mlflow_model/ -- an MLflow model directory that records the exact runtime environment
   (conda.yaml / requirements.txt / python_env.yaml). Use the matching flavor, e.g.
   mlflow.sklearn.save_model(model, path="mlflow_model") (or mlflow.xgboost /
   mlflow.lightgbm / mlflow.pyfunc).
3. metrics.json -- task type, chosen library+model, the target column, and evaluation
   metrics (R2/RMSE for regression, accuracy/F1 for classification, silhouette for
   clustering).

When the Code Interpreter produces files, name each one explicitly in your final answer
so the file can be downloaded. End with a concise summary: task type, library + model, key
metrics, the target column, and the names of the exported files.
"""


# The Copilot CLI requests permission per capability "kind": shell, read, write, mcp, url.
# The operator chooses which kinds to allow via COPILOT_ALLOWED_PERMISSIONS (comma-separated);
# every other kind is denied. For this agent the default is "mcp,read": the agent must use the
# toolbox (mcp) and may read files handed to it, but local code execution (shell) and local
# file writes (write) are deliberately withheld so execution stays in the toolbox sandbox.
# FOUNDRY_* / AGENT_* env names are reserved by the platform, hence the COPILOT_ prefix.
DEFAULT_ALLOWED_PERMISSIONS = "mcp,read"
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


def build_toolbox_mcp_servers() -> dict[str, MCPServerConfig]:
    """Wire the Foundry Toolbox's MCP endpoint into the Copilot harness as an HTTP MCP server.

    The toolbox endpoint is taken from TOOLBOX_ENDPOINT, or constructed from the project
    endpoint + TOOLBOX_NAME. The harness only supports a *static* Authorization header, so a
    bearer token is minted once here with the container's managed identity (locally, your
    `az login`). That token lives roughly an hour; a very long-lived container may need a
    restart to re-mint it. A per-request refresh would require the Agent-Framework
    `FoundryToolbox` tool rather than the harness's static MCP config.
    """
    endpoint = os.getenv("TOOLBOX_ENDPOINT")
    if not endpoint:
        project = os.getenv("FOUNDRY_PROJECT_ENDPOINT") or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        name = os.getenv("TOOLBOX_NAME")
        if not project or not name:
            raise RuntimeError(
                "Toolbox endpoint is not configured. Set TOOLBOX_ENDPOINT, or set both "
                "TOOLBOX_NAME and a project endpoint (FOUNDRY_PROJECT_ENDPOINT / "
                "AZURE_AI_PROJECT_ENDPOINT)."
            )
        endpoint = f"{project.rstrip('/')}/toolboxes/{name}/mcp?api-version=v1"

    # *.services.ai.azure.com data-plane calls use the ai.azure.com scope.
    scope = os.getenv("TOOLBOX_TOKEN_SCOPE", "https://ai.azure.com/.default")
    with SyncDefaultAzureCredential() as credential:
        token = credential.get_token(scope).token

    print(f"Toolbox MCP endpoint: {endpoint}", flush=True)
    return {
        "foundry-toolbox": {
            "type": "http",
            "url": endpoint,
            "tools": ["*"],
            "headers": {"Authorization": f"Bearer {token}"},
        }
    }


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
            mcp_servers=build_toolbox_mcp_servers(),
            timeout=1800,
        ),
    )

    # history_source="agent": GitHubCopilotAgent is a custom SupportsAgentRun agent, so it
    # manages its own conversation history. The default "agent_server" expects a RawAgent and
    # the container crashes at startup (invoke returns HTTP 424 session_not_ready).
    server = ResponsesHostServer(agent, history_source="agent")
    server.run()


if __name__ == "__main__":
    main()
