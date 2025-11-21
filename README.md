# answer_reviewer

A set of FastAPI services that review and, when needed, rewrite answers to product questions in Portuguese or Spanish using multi‑agent LLM workflows. Each service variant exposes the same endpoints but uses a different orchestration strategy.

## Repository layout
- `user_reviewer/`: Two‑agent loop where a reviewer scores the answer and a user proxy rewrites it until the score is good enough.
- `group_chat/`: Reviewer → Rewriter → Evaluator agents coordinated by a group chat manager.
- `swarm/`: Swarm/Autogen pattern with semantic reviewer, contextual reviewer, suggester, rewriter, and decider; captures richer scoring and decision data.
- `tests/`: Sample data and a helper script (`jsonl_to_csvs.py`) for slicing JSONL datasets into JSON chunks.
- `requirements.txt`: Python dependencies.

## Request model (all services)
`RevisionRequest` fields:
- `id`: int
- `question`: str
- `answer`: str
- `correct`: bool
- `feedback`: str | null
- `locale`: str (`pt` → Portuguese, anything else → Spanish)
- `intent`: object (expects at least `name`, may include `confidence`)
- `context`: object
- `metadata`: array
- `category`: str

Example payload:
```json
{
  "id": 1,
  "question": "Qual o prazo de entrega?",
  "answer": "Chega em até 5 dias úteis.",
  "correct": false,
  "feedback": null,
  "locale": "pt",
  "intent": { "name": "delivery_time", "confidence": 0.92 },
  "context": { "shipping_time": "5 dias úteis" },
  "metadata": [],
  "category": "shipping"
}
```

## Service variants

### `user_reviewer` (fast feedback loop)
- Agents: `Reviewer` (scores 0–10 with per‑aspect tags) and `User` proxy (rewrites using the suggestions and stops when the score passes 7).
- Response: `{"response": "<final_answer>"}`. If no rewrite was possible, the reviewer can return `"It is not possible to provide a revised answer."`.
- Persistence: Writes `results.csv` with original/revised answers, scores, suggestions, intent, category, and cost (if the LLM reports it).

### `group_chat` (reviewer → rewriter → evaluator)
- Agents: `Reviewer` (scores 0–10 + suggestions), `Rewriter` (produces `<revised_answer>` or `THIS QUESTION CANNOT BE ANSWERED!!`), `Evaluator` (chooses final answer, emits `<new_score>`).
- Response: `{"final_answer": "...", "previous_score": <int|null>, "new_score": <int|string>}`. Answers with low revised scores or flagged as unanswerable become `DO_NOT_ANSWER`.
- Persistence: Appends to `results.csv` with both scores, suggestions, revised answer, and final decision.

### `swarm` (semantic + contextual + decision loop)
- Agents: Semantic reviewer (0–5), Contextual reviewer (0–5), Suggester, Rewriter, Decider. Uses `autogen` swarm `DefaultPattern` with function calls to pass scores and state.
- Decision rules: If the combined new score ≤ 7, or the decider returns `REWRITE`/`DO_NOT_ANSWER`, the final answer is `DO_NOT_ANSWER`; if the original score > 7, the original answer is retained.
- Response: Same shape as `group_chat`.
- Persistence: `results.csv` includes original/revised scores, suggestions, number of revisions, decision, and justification.

## Running a service
1) Install dependencies (Python 3.10+ recommended):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure LLM access:
- `user_reviewer` and `group_chat` use OpenAI `gpt-4o`; set `OPENAI_API_KEY` in a `.env` or environment.
- `swarm` is configured for a local Ollama endpoint (`qwen3:8b` at `http://localhost:11434/v1`); adjust `config_list` in `swarm/agents/agents.py` if needed.

3) Run one of the apps (each exposes `/revise` and `/revise-questions`):
```bash
# User-driven reviewer/rewriter
uvicorn user_reviewer.main:app --reload --port 8000

# Group chat reviewer → rewriter → evaluator
uvicorn group_chat.main:app --reload --port 8001

# Swarm-based multi-agent pipeline
uvicorn swarm.main:app --reload --port 8002
```

## API usage
- `POST /revise`: single `RevisionRequest`. Returns the final answer (and scores for `group_chat`/`swarm`).
- `POST /revise-questions`: array of `RevisionRequest` objects. Returns a list of per-item responses.

All services append a row to `results.csv` in the working directory after each request.
