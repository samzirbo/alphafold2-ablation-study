# AlphaFold2 Ablation Study

Ablation studies on AlphaFold2 inputs (query sequence and MSA) for sampling alternative protein conformations.

**TUM SoSe2026 | Chair of Bioinformatics**

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Dependencies are declared in `pyproject.toml` and pinned exactly in `uv.lock`.

### 1. Install uv

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal, then verify:

```bash
uv --version
```

### 2. Install dependencies

```bash
git clone <repo-url>
cd alphafold2-ablation-study
uv sync
```

This creates a `.venv`, installs the exact versions from `uv.lock`, and is fully reproducible across machines.

### 3. Activate the environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
.venv\Scripts\activate.bat
```

Or skip activation entirely and prefix commands with `uv run`:

```bash
uv run python my_script.py
uv run jupyter notebook
```

### 4. Verify

```bash
uv run python -c "import tmtools; import prody; import Bio; print('All good!')"
```

---

## Managing Dependencies

> **Always use `uv add` / `uv remove` -- never `uv pip install`.**
>
> `uv pip install` installs into the venv but does NOT update `pyproject.toml` or `uv.lock`, so the dependency would be invisible to everyone else.

```bash
# Add a new dependency (updates pyproject.toml + uv.lock + installs)
uv add <package>

# Remove a dependency
uv remove <package>

# After pulling changes that modified pyproject.toml or uv.lock
uv sync
```

---
