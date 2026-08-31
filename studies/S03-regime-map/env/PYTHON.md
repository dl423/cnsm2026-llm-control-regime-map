# Python environment

- CPython 3.12.13
- uv 0.12.1
- exact package versions: `requirements.txt`

From the repository root:

```bash
uv venv --python 3.12.13 .venv
. .venv/bin/activate
uv pip sync studies/S03-regime-map/env/requirements.txt
```

The package file captures the original environment, including analysis and
experiment dependencies. It is intentionally not minimized after the campaign.
