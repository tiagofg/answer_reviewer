# Answer Reviewer

An experimental comparison of three multi-agent approaches to reviewing and revising answers to product questions. Each approach is exposed through a small FastAPI application and evaluates Portuguese or Spanish answers against the supplied question, intent, category, context, and metadata.

The implementations share an API schema while differing in agent roles, orchestration, scoring, and return values. See [Approaches and implementation notes](docs/approaches.md) for a source-level comparison.

## Repository structure

| Path | Approach | Configured model |
| --- | --- | --- |
| [`user_reviewer/`](user_reviewer/) | Reviewer and automated user-proxy revision loop | OpenAI `gpt-4o` |
| [`group_chat/`](group_chat/) | Round-robin reviewer, rewriter, and evaluator | OpenAI `gpt-4o` |
| [`swarm/`](swarm/) | Routed semantic reviewer, contextual reviewer, suggester, rewriter, and decider | Ollama `qwen3:8b` |
| [`requirements.txt`](requirements.txt) | Shared, pinned Python dependencies | — |

## Setup

Python 3.11 is a practical starting point for the pinned numerical dependencies. It is not a tested support matrix for this repository.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For `user_reviewer` and `group_chat`, set `OPENAI_API_KEY` in the environment or a root-level `.env` file.

For `swarm`, run an Ollama-compatible API at `http://localhost:11434/v1` with the `qwen3:8b` model available. That endpoint and model are set directly in [`swarm/agents/agents.py`](swarm/agents/agents.py).

## Run one approach

Activate the root virtual environment first, then start an app from its own directory. The modules use top-level `models`, `services`, and `agents` imports, so the working directory matters.

```bash
(cd user_reviewer && python -m uvicorn main:app --reload --port 8000)
(cd group_chat && python -m uvicorn main:app --reload --port 8001)
(cd swarm && python -m uvicorn main:app --reload --port 8002)
```

Each command starts a separate service. FastAPI's interactive API documentation is available at `/docs` on the selected port.

## API

All three apps accept the same request object:

```json
{
  "id": 1,
  "question": "Qual é o prazo de entrega?",
  "answer": "A entrega leva cinco dias úteis.",
  "correct": false,
  "feedback": null,
  "locale": "pt",
  "intent": { "name": "delivery_time" },
  "context": { "delivery_time": "5 dias úteis" },
  "metadata": [],
  "category": "shipping"
}
```

`locale == "pt"` selects Portuguese; every other value is treated as Spanish. The services read the intent from `intent.name`.

### `POST /revise`

The `user_reviewer` app wraps its final string:

```json
{ "response": "A entrega leva cinco dias úteis." }
```

The `group_chat` and `swarm` apps return an object:

```json
{
  "final_answer": "A entrega leva cinco dias úteis.",
  "previous_score": 8,
  "new_score": "-"
}
```

Scores may be `null` when parsing or registration does not produce one. A string `"-"` is used for a missing new score; `group_chat` and `swarm` also return `"-"` as `final_answer` when their internal decision is `DO_NOT_ANSWER`.

### `POST /revise-questions`

Send an array of request objects. Every implementation returns an outer object with a `responses` array:

```json
{ "responses": [] }
```

For `user_reviewer`, each item in `responses` is a string. For `group_chat` and `swarm`, each item is an object with `final_answer`, `previous_score`, and `new_score`.

## Operational notes

Each request appends a row to `results.csv` in the app's working directory. The services do not isolate concurrent CSV writes. Their agent objects are also created at module import time and reused between requests; `group_chat` reads the shared manager's chat history, and `swarm` clears and repopulates one module-level context object. Concurrent calls, and potentially accumulated group-chat history, can therefore interfere with result extraction.

The swarm decision checks use Python identity comparisons for some string values (`is` rather than equality). Their behavior is therefore implementation-dependent and may not match the intended decision text consistently.

This repository contains no automated tests, CI configuration, bundled evaluation dataset, authentication, or deployment configuration. Treat it as experimental code for inspecting and comparing orchestration approaches.
