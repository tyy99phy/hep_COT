# hep-copilot

**HEP Paper Literature Reasoning Copilot + Comparative CoT Study**

A tool-using LLM agent that reads HEP experimental papers (arXiv LaTeX
source + figures + HEPData + INSPIRE reference graph) and answers
targeted questions about systematics, statistical methods, limits, and
analysis strategy with verifiable citations.

Also ships a **comparative reasoning study** harness: run a paper QA
set through a *teacher* (GPT-5.4, reasoning summaries) and a *student*
(DeepSeek-reasoner V3.2, raw CoT), then use structured answers to
measure the reasoning gap, with optional autoloop refinement of the
student until the hard correctness (L1) criterion is met.

## Install

```bash
pip install -e .
# requires openai>=1.50 and httpx>=0.27; pyyaml for HEPData parsing
```

Set provider API keys:

```bash
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=sk-...
# optional:
export OPENAI_BASE_URL=https://...      # for proxies
export DEEPSEEK_BASE_URL=https://...    # override default https://api.deepseek.com
```

## Quick start — interactive copilot

```bash
hep-copilot paper 2206.08956 --provider openai
# or
hep-copilot paper 2206.08956 --provider deepseek
```

```
You > What are the top-3 dominant systematic uncertainties?
[thinking] need to locate the systematics section and a breakdown table...
[tool] get_paper_section(section_pattern="Systematic")
[tool] fetch_hepdata(arxiv_id="2206.08956")
[tool] get_hepdata_table(arxiv_id="2206.08956", table_name="Systematic breakdown")
[answer] The three dominant systematics are (1) JES at 3.2% [§6.2, HEPData/Tbl 5] ...
```

Type `/tools` to list available tools, `/paper <id>` to switch paper,
`/provider <name>` to switch backend mid-session, `/save` to persist.

## One-shot ask

```bash
hep-copilot ask 2206.08956 "What statistical framework does the paper use?"
```

## Comparative study

```bash
hep-copilot study 2206.08956 \
  --questions study_inputs/qa_seed_syst_stat_limit.json \
  --teacher openai \
  --student deepseek \
  --autoloop --max-rounds 5 \
  --output-dir ./study_sessions
```

Output:
```
study_sessions/teacher_openai_2206.08956_<ts>.jsonl
study_sessions/student_native_deepseek_2206.08956_<ts>.jsonl
study_sessions/student_aligned_deepseek_2206.08956_<ts>.jsonl
study_sessions/metrics_2206.08956_<ts>.json
```

The `metrics_*.json` aggregates per-question and per-category:
  * Native L1 pass rate (student's first unassisted answer)
  * Aligned convergence rate (student after ≤ N feedback rounds)
  * Tool-call sequence edit distance (teacher vs student)
  * Number / citation hit rates

## Architecture

```
hep_cot/
├── llm/                   # Provider abstraction (OpenAI Responses + DeepSeek-reasoner)
├── agent/                 # Model ↔ tool loop + state + tool registry
├── tools/                 # HEP tools: paper / figures / INSPIRE / HEPData / arxiv
├── session/               # Interactive REPL + persistence bridge
├── prompts/               # Copilot / teacher / student system prompts
├── study/                 # Comparative study: teacher / student / autoloop / judge / metrics
├── paper_fetcher.py       # (existing) arXiv/INSPIRE → tex
├── tex_extractor.py       # (existing) LaTeX source → clean tex
├── figure_extractor.py    # (existing) LaTeX figures → PNG + captions
├── cot_store.py           # (existing) session persistence
└── cli.py                 # hep-copilot / hep-cot entry
```

Provider emits a unified event stream:

```
ThinkingDelta  (reasoning, raw for DeepSeek / summary for GPT-5.4)
TextDelta      (final answer tokens)
ToolCall       (function-call request)
TurnEnd        (stop reason + usage)
```

The agent loop runs model → tool-execute → model until `stop_reason == "stop"`.

## Tools (MVP-1)

| Tool                 | Data source          |
|----------------------|----------------------|
| `fetch_paper`        | arXiv LaTeX source   |
| `get_paper_info`     | local cache          |
| `get_paper_section`  | local tex            |
| `search_text`        | local tex (regex)    |
| `get_bibliography`   | local .bbl           |
| `list_figures`       | LaTeX figures        |
| `get_figure`         | LaTeX + PNG          |
| `inspire_references` | INSPIRE-HEP API      |
| `inspire_citations`  | INSPIRE-HEP API      |
| `arxiv_search`       | arXiv API            |
| `fetch_hepdata`      | HEPData API          |
| `get_hepdata_table`  | HEPData API          |

All external API calls are file-cached under `--cache-dir`.

## Legacy Codex mode

The original Codex App-Server paper-analysis flow is preserved under
`legacy-*` subcommands:

```bash
hep-copilot legacy-chat
hep-copilot legacy-paper 2206.08956
hep-copilot legacy-batch tasks.json
hep-copilot legacy-export session.json -o session.md
```

## Testing

```bash
pytest tests/
```

Offline smoke tests cover the agent loop, tool registry, judge, and
paper_tools against the cached `2206.08956` fixture.

## Roadmap

- M2: L3 CoT embedding similarity in judge; HEP reinterpretation workflow
- M3: multi-paper cross-referencing; evaluation benchmark publication

## License / credits

See repository root.
