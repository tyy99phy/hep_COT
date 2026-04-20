# hep_cot

**HEP paper literature reasoning copilot + comparative CoT study harness**

A tool-using LLM agent that reads HEP experimental papers (arXiv LaTeX
source + figures + HEPData + INSPIRE reference graph) and answers
targeted questions about systematics, statistical methods, limits, and
analysis strategy with verifiable citations.

Also ships a **protocol-driven comparative reasoning study** harness: run
a paper through a *teacher* (e.g. GPT with reasoning summaries) and a
*student* (e.g. DeepSeek-reasoner with raw CoT) across six predefined
analysis phases, then measure the reasoning gap, with optional autoloop
refinement of the student until a per-phase hard-correctness (L1)
criterion is met.

---

## Prerequisites

- Python **3.10+**
- Outbound HTTPS to `arxiv.org`, `inspirehep.net`, and `www.hepdata.net`
  (only needed once per paper — results are cached under `paper_cache/`).
- At least one LLM provider API key:
  - `OPENAI_API_KEY` for the OpenAI Responses backend (teacher / default
    copilot).
  - `DEEPSEEK_API_KEY` for DeepSeek-reasoner (student).

## Install

```bash
git clone https://github.com/tyy99phy/hep_COT.git
cd hep_COT
pip install -e .
# optional: PDF export for the `export` subcommand
pip install -e ".[pdf]"
```

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
$EDITOR .env
# then either `source .env` or use direnv / python-dotenv
```

## Bootstrap demo fixtures

Downloads the default demo paper's LaTeX source into `paper_cache/` so
that the Quick start examples have something to read and the offline
unit tests can run end-to-end:

```bash
python scripts/bootstrap_fixtures.py            # arXiv:2206.08956
python scripts/bootstrap_fixtures.py 1811.10461 # any other arXiv id
```

Re-running is a no-op once the source is cached.

## Quick start — interactive copilot

```bash
hep-cot paper 2206.08956 --provider openai
# or
hep-cot paper 2206.08956 --provider deepseek
```

```
You > What are the top-3 dominant systematic uncertainties?
[thinking]   need to locate the systematics section and a breakdown table...
[tool]       get_paper_section(section_pattern="Systematic")
[tool]       fetch_hepdata(arxiv_id="2206.08956")
[tool]       get_hepdata_table(arxiv_id="2206.08956", table_name="Systematic breakdown")
[answer]     The three dominant systematics are (1) JES at 3.2% [§6.2, HEPData/Tbl 5] ...
```

REPL commands: `/tools` lists available tools, `/paper <id>` switches
paper, `/provider <name>` switches backend, `/save` persists the
session.

## One-shot ask

```bash
hep-cot ask 2206.08956 "What statistical framework does the paper use?"
```

## Comparative study (teacher vs student, six analysis phases)

```bash
hep-cot study 2206.08956 \
  --teacher openai \
  --student deepseek \
  --autoloop \
  --max-rounds-per-phase 3 \
  --output-dir ./study_sessions
```

Output layout:

```
study_sessions/teacher_<id>_<ts>/           # teacher per-phase JSONs
study_sessions/student_native_<id>_<ts>/    # student first-pass JSONs
study_sessions/student_aligned_<id>_<ts>/   # autoloop-refined JSONs (if --autoloop)
study_sessions/metrics_<id>_<ts>.json       # aggregated pass-rate metrics
```

`metrics_*.json` aggregates per-phase and overall:

- **Native L1 pass rate** — student's first unassisted answer vs teacher gold.
- **Aligned convergence rate** — student pass rate after ≤ N feedback rounds.
- **Tool-call sequence edit distance** — teacher vs student tool-use trajectory.
- **Number / citation hit rates** — derived from the structured-answer schema.
- **Figure coverage** — fraction of paper figures referenced at least once.

Re-render a previously completed run into `ANALYSIS.md` / `ANALYSIS.pdf`
without re-calling any LLM:

```bash
hep-cot export ./study_sessions/teacher_<id>_<ts>
hep-cot export ./study_sessions                       # bulk re-export
```

## Architecture

See [docs/pipeline_overview.pdf](./docs/pipeline_overview.pdf) for the
full pipeline diagram.

```
src/hep_cot/
├── llm/                   # Provider abstraction (OpenAI Responses + DeepSeek-reasoner)
├── agent/                 # Model ↔ tool loop + state + tool registry
├── tools/                 # HEP tools: paper / figures / INSPIRE / HEPData / arXiv
├── session/               # Interactive REPL + persistence + MD/PDF export
├── prompts/               # Copilot / teacher / student system prompts
├── study/                 # Protocol phases, teacher, student, autoloop, judge, metrics
├── paper_fetcher.py       # arXiv / INSPIRE → LaTeX source + metadata
├── tex_extractor.py       # LaTeX source → clean text
├── figure_extractor.py    # LaTeX figures → PNG + captions
├── cot_store.py           # session persistence
└── cli.py                 # `hep-cot` / `hep-copilot` entry
```

Each provider emits a unified event stream:

```
ThinkingDelta  (reasoning — raw for DeepSeek, summary for OpenAI)
TextDelta      (final-answer tokens)
ToolCall       (function-call request)
TurnEnd        (stop reason + usage)
```

The agent loop runs `model → execute-tool → model` until
`stop_reason == "stop"`.

## Tools (MVP-1)

| Tool                 | Data source          |
|----------------------|----------------------|
| `fetch_paper`        | arXiv LaTeX source   |
| `get_paper_info`     | local cache          |
| `get_paper_section`  | local tex            |
| `search_text`        | local tex (regex)    |
| `get_bibliography`   | local `.bbl`         |
| `list_figures`       | LaTeX figures        |
| `get_figure`         | LaTeX + PNG          |
| `inspire_references` | INSPIRE-HEP API      |
| `inspire_citations`  | INSPIRE-HEP API      |
| `arxiv_search`       | arXiv API            |
| `fetch_hepdata`      | HEPData API          |
| `get_hepdata_table`  | HEPData API          |

All external API calls are file-cached under `--cache-dir` (default
`./paper_cache`).

## Legacy Codex mode

The original Codex App-Server paper-analysis flow is preserved under
`legacy-*` subcommands for reproducibility of earlier experiments:

```bash
hep-cot legacy-chat
hep-cot legacy-paper 2206.08956
hep-cot legacy-batch tasks.json
hep-cot legacy-export session.json -o session.md
```

## Testing

```bash
pytest tests/
```

Offline smoke tests cover the agent loop, tool registry, judge, and
paper tools. The paper-tool test auto-skips if the demo fixture is
absent — run `python scripts/bootstrap_fixtures.py` once first to
enable it.

## Citing

A [`CITATION.cff`](./CITATION.cff) is provided for GitHub's "Cite this
repository" button. BibTeX / plain-text exports are available from the
same menu.

## License

[MIT](./LICENSE) — see `LICENSE` for full text.
