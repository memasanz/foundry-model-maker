FROM python:3.12-slim

WORKDIR /app

# Install Node.js and the GitHub Copilot CLI, which powers GitHubCopilotAgent.
# The `copilot` binary is spawned by the agent at runtime. Node.js 22+ is required.
# No OpenMP / ML native libs are needed here because code runs in the Foundry Toolbox
# Code Interpreter sandbox, not in this container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @github/copilot \
    && apt-get purge -y --auto-remove gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . user_agent/
WORKDIR /app/user_agent

RUN if [ -f requirements.txt ]; then \
        pip install -r requirements.txt; \
    else \
        echo "No requirements.txt found"; \
    fi

# Override: agent-framework-github-copilot==1.0.2 pins github-copilot-sdk==1.0.2,
# which lacks `bearer_token_provider` (the OAuth callback needed for Foundry BYOK).
# 1.0.11 adds it and is runtime-compatible with af-github 1.0.2. pip emits a
# dependency-conflict warning but leaves 1.0.11 installed, which is what we want.
# Remove once af-github depends on github-copilot-sdk>=1.0.11.
RUN pip install "github-copilot-sdk==1.0.11"

EXPOSE 8088

# Precompile Python bytecode at build time so cold starts don't pay for it.
# These base images ship almost none of the standard library precompiled, so without
# this every new container recompiles hundreds of stdlib modules on first import.
# Dependencies are best-effort (`|| true`) so a stray unparsable file vendored by a
# third-party package can't fail the build; application code is compiled strictly so a
# syntax error surfaces at build time rather than at container start.
# PYTHONDONTWRITEBYTECODE is cleared for these commands only: it blocks bytecode
# *writes*, not *reads*, so images that set it still benefit.
RUN PYTHONDONTWRITEBYTECODE= python -m compileall -q $(python -c "import sysconfig as s; print(s.get_paths()['stdlib'], s.get_paths()['purelib'])") || true; \
    PYTHONDONTWRITEBYTECODE= python -m compileall -q .

CMD ["python", "main.py"]
