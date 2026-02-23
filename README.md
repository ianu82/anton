# Anton

```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀          ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐            █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
```


Anton is an advanced AI coworker. You tell it what you need done and it figures out the rest. If it needs code, it writes it. If it needs a tool it doesn't have, it builds one. If it needs to run five things in parallel, it spawns minions.

## Quick start

**macOS / Linux:**
```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh
```

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

Open a new terminal, then:
```
anton
```

That drops you into a conversation. Talk to Anton like a person.

Once running try asking anton anything, for example this:

```
you> Information about inflation in the US is found on this website,
     https://www.bls.gov/news.release/cpi.nr0.htm — plot me the CPI
     items stacked per month.
```

What happens next is the interesting part. Anton doesn't have any particular skill to begin with. It figures it out live: fetches the page, parses the HTML, writes scratchpad code on the fly, and generates a stacked bar chart with the information you asked for — all in one conversation, with no setup.
That's the point: you describe a problem in plain language, and Anton assembles the toolchain, writes the code, and delivers the result.

## How it works

Anton is a self-evolving autonomous system that collaborates with people to solve problems.

Anton operates in two modes depending on what you need:

**Chat + Scratchpad** — For most tasks, Anton works directly in conversation. It thinks through the problem, writes and runs Python in a persistent scratchpad (variables, imports, and data survive across steps), installs packages as needed, and iterates until it has your answer. 

**Autonomous pipeline** — For larger tasks that need structured execution, Anton switches to a four-phase pipeline:

```
Task → Memory Recall → Planning → Skill Building (if needed) → Execution
```

**Explainable by default** — You can always ask Anton to explain what it did. Ask it to dump its scratchpad and you get a full notebook-style breakdown: every cell of code it ran, the outputs, and errors — so you can follow its reasoning step by step.

## Scratchpad

Anton includes a persistent Python scratchpad — a notebook-style environment it drives programmatically. This makes Anton particularly great at **data analysis tasks**: counting, parsing, transforming, aggregating, and exploring datasets with real computation instead of LLM guesswork.

When you ask Anton to analyze data, it writes and executes Python in the scratchpad, building up state across cells exactly like a Jupyter notebook. Variables, imports, and intermediate results persist across steps. When you ask "how did you solve that?", it dumps a clean notebook-style summary of its work — code blocks, truncated output samples, error summaries — so you can follow the reasoning without wading through raw logs.

What the scratchpad handles well:
- **Data analysis** — Load a CSV, filter rows, compute aggregates, pivot tables, plot distributions
- **Text processing** — Parse logs, extract patterns, count tokens, transform formats
- **Math and counting** — Character counts, statistical calculations, combinatorics
- **Multi-step exploration** — Build up understanding incrementally, inspect intermediate results
- **LLM-powered computation** — Call `get_llm()` inside scratchpad code for AI-assisted analysis (classification, extraction, summarization) over your data

The scratchpad also has access to Anton's skills (`run_skill()`) and an `agentic_loop()` for multi-step AI workflows — so it can orchestrate tool-calling LLM loops right inside the notebook.

## Minions

Minions are background workers. Anton handles the conversation, but when a task is long-running, independent, or recurring, it spawns a minion to handle it separately.

### Spawning

In chat, Anton recognizes work that should run in the background — large batch jobs, monitoring tasks, periodic checks — and spawns a minion. From the CLI:

```bash
anton minion "check email and flag urgent items" --every 30m
anton minion "run test suite" --max-runs 1
anton minion "monitor API health" --every 5m --start 2025-01-15T09:00 --end 2025-01-15T17:00
```

Each minion gets its own directory (`.anton/minions/<id>/`) with a status file, session transcript, and any output artifacts. If scheduling options aren't passed via CLI, the minion can ask conversationally.

### Communication

Minions write their results to `.anton/minions/<id>/` — status.json, transcripts, and artifacts. The parent Anton reads these to check progress. No sockets, no IPC — just the filesystem. When a minion completes, its summary is available for the next conversation turn.

### Scheduling

Minions support flexible scheduling:
- `--every` — Repeat frequency (`5m`, `1h`, `30s`)
- `--start` / `--end` — Time window
- `--max-runs` — Cap on total executions
- `cron_expr` — Standard cron expressions for advanced scheduling

### Lifecycle

Statuses: `pending → running → completed | failed | killed`

Anton decides which minions to kill. When a minion is killed, its schedule is also removed — no orphaned jobs. Minions track their own run count and respect max_runs limits.

**Current state.** Data models (`MinionInfo`, `MinionRegistry`), CLI scaffolding, directory creation, and status persistence exist. Process spawning and real-time monitoring are not yet implemented.

## Commands

| Command | What it does |
|---|---|
| `anton` | Interactive chat |
| `anton run "task"` | Execute a task autonomously |
| `anton --folder /path` | Use a specific workspace |
| `anton skills` | List discovered skills |
| `anton sessions` | Browse past work |
| `anton learnings` | What Anton has learned |
| `anton minion "task"` | Spawn a background worker |
| `anton minion "task" --every 5m` | Spawn a recurring minion |
| `anton minions` | List tracked minions |

## Workspace

When you run `anton` in a directory, it checks for an `anton.md` file. If the folder has existing files but no `anton.md`, Anton asks before setting up — it won't touch your stuff without permission.

Once initialized, the workspace looks like:

```
project/
├── anton.md              # Project context (read every conversation)
└── .anton/
    ├── .env              # Secrets (API keys, tokens — never pass through LLM)
    ├── context/          # Self-awareness files (project facts, conventions)
    ├── skills/           # User and auto-generated skills
    ├── sessions/         # Task transcripts and summaries
    ├── learnings/        # Extracted insights
    └── minions/          # One folder per minion (<id>/status.json, artifacts)
```

**anton.md** — Write anything here. Project context, conventions, preferences. Anton reads it at the start of every conversation.

**Secret vault** — When Anton needs an API key or token, it asks you directly and stores the value in `.anton/.env`. The secret never passes through the LLM — Anton just gets told "the variable is set."

All data lives in `.anton/` in the current working directory. Override with `anton --folder /path`.


## Configuration

```
.anton/.env              # Workspace-local secrets and API keys
ANTON_ANTHROPIC_API_KEY  # Anthropic API key
ANTON_PLANNING_MODEL     # Model for planning (default: claude-sonnet-4-6)
ANTON_CODING_MODEL       # Model for coding (default: claude-opus-4-6)
```

Env loading order: `cwd/.env` → `.anton/.env` → `~/.anton/.env`

## Manual install

If you already have [uv](https://docs.astral.sh/uv/):
```
uv tool install git+https://github.com/mindsdb/anton.git
```

## Upgrade / Uninstall

```
uv tool upgrade anton
uv tool uninstall anton
```

### Prerequisites

- **git** — required ([macOS](https://git-scm.com/downloads/mac) / `sudo apt install git` / `winget install Git.Git`)
- **Python 3.11+** — optional (uv downloads it automatically if missing)
- **curl** — macOS/Linux only, usually pre-installed
- Internet connection. No admin/sudo required.

## Is "Anton" a Mind?

Yes, at mindsDB we build AI systems that collaborate with people to accomplish tasks, inspired by the culture series books, so yes, Anton is a Mind :)

## Why the name "Anton"?

We really enjoyed the show *Silicon Valley*. Gilfoyle's AI — Son of Anton — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We thought it was was great name for an AI that can learn on its own, so we kept Anton, dropped the "Son of".

## License

MIT
