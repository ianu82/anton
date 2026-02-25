# Architecture

```
anton/
├── cli.py                 # Typer CLI (chat, setup, service, eval)
├── chat.py                # Multi-turn conversation with tool delegation
├── tools.py               # Tool schemas + dispatchers (scratchpad, connector, etc.)
├── scratchpad.py          # Persistent Python scratchpad manager
├── scratchpad_boot.py     # Scratchpad subprocess bootstrap/runtime helpers
├── workspace.py           # Workspace init, anton.md, secret vault
├── connectors/            # Connector adapter layer (HTTP and local SQLite)
├── governance/            # Policy + budget primitives
├── service/
│   ├── app.py             # FastAPI app and endpoint wiring
│   ├── runtime.py         # Session runtime manager + run governance hooks
│   ├── memory.py          # Retrieved memory context builder for service turns
│   ├── skills.py          # Prompt template parsing/rendering for skill and schedule runs
│   ├── store.py           # SQLite persistence + audit events
│   └── eval.py            # Benchmark/evaluation harness
├── llm/
│   ├── client.py          # Planning vs coding model abstraction
│   ├── provider.py        # Provider interfaces and stream events
│   ├── anthropic.py       # Anthropic provider implementation
│   └── openai.py          # OpenAI provider implementation
├── context/
│   └── self_awareness.py  # .anton/context/ files injected into prompts
├── memory/
│   ├── store.py           # Session transcript store
│   └── learnings.py       # Learnings index store
└── channel/
    ├── branding.py        # Banner/dashboard rendering
    └── theme.py           # Console theme utilities
```
