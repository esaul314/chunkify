# Dependency Policy

`pyproject.toml` is the dependency source of truth. It declares the supported
runtime, development, and optional integration environments; `requirements.txt`
is retained only as a compatibility entry point for older setup instructions and
delegates to the editable base package install.

## Installation profiles

| Profile | Install command | Purpose |
| --- | --- | --- |
| Base | `pip install -e .` | PDF, EPUB, and plain-text conversion |
| Development | `pip install -e ".[dev]"` | Tests, formatting, linting, typing, and nox |
| Enrichment | `pip install -e ".[enrichment]"` | Optional `ai_enrich` / LiteLLM integration |
| PDF fallback | `pip install -e ".[pdf-fallback]"` | Optional `pdfminer.six` fallback engine |
| PDF enhancement | `pip install -e ".[pdf-enhancement]"` | Optional PyMuPDF4LLM enhancement |

EPUB parsing dependencies are part of the base runtime because `.epub` is a
supported input format. AI enrichment, the pure-Python PDF fallback, and the
PyMuPDF4LLM enhancement remain opt-in so base conversion stays usable without
those integrations.

The PDF `pdftotext` fallback also requires the system `poppler-utils` package;
it is not a Python dependency. When neither optional fallback is available, the
primary PyMuPDF extraction path remains the default.

## Reproducibility

The project currently uses version ranges in `pyproject.toml` rather than
committing a machine-specific lockfile. The local `pdf-env` and any generated
`pip freeze` output are environment artifacts, not repository policy. If release
or CI reproducibility later requires exact resolution, generate a constraints
file from the declared profile in a controlled environment and review it as a
separate artifact; do not hand-edit a second dependency list.
