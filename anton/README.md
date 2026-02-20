# Architecture

```
anton/
├── cli.py              # Typer CLI: chat, run, skills, sessions, minions
├── chat.py             # Multi-turn conversation with tool-call delegation
├── workspace.py        # Workspace init, anton.md, secret vault
├── core/
│   ├── agent.py        # Orchestrator: memory → plan → build → execute
│   ├── planner.py      # LLM-powered task decomposition
│   ├── executor.py     # Step-by-step plan execution
│   └── estimator.py    # Duration tracking for ETAs
├── skill/
│   ├── base.py         # @skill decorator and SkillResult
│   ├── registry.py     # Discovery and lookup
│   ├── builder.py      # LLM-powered skill generation
│   └── loader.py       # Dynamic module loading
├── memory/
│   ├── store.py        # Session history (JSONL transcripts)
│   ├── learnings.py    # Extracted insights indexed by topic
│   └── context.py      # Builds memory context for the planner
├── context/
│   └── self_awareness.py  # .anton/context/ files injected into every LLM call
├── minion/
│   └── registry.py     # Minion data models and lifecycle
├── llm/
│   ├── client.py       # Planning vs coding model abstraction
│   ├── prompts.py      # System prompts
│   └── anthropic.py    # Anthropic provider
├── channel/            # Output rendering (terminal, future: slack, etc.)
└── events/             # Async event bus for status updates
```
