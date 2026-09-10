# Approaches and implementation notes

The three services accept the same `RevisionRequest`, but their conversations and result handling differ. This page describes the behavior implemented in each `agents/agents.py` and `services/revision_service.py` pair.

## Comparison

| Characteristic | `user_reviewer` | `group_chat` | `swarm` |
| --- | --- | --- | --- |
| Roles | Reviewer, automated user proxy | Reviewer, rewriter, evaluator, user proxy, group-chat manager | Semantic reviewer, contextual reviewer, suggester, rewriter, decider, user proxy |
| Orchestration | Two-agent chat; the proxy revises and resubmits | Round-robin group chat managed by AG2 | `DefaultPattern` with tool-return targets and shared context variables |
| Model | OpenAI `gpt-4o` | OpenAI `gpt-4o` | Ollama `qwen3:8b` through an OpenAI-compatible local endpoint |
| Review score | Semantic 0–5 + contextual 0–5, parsed as a 0–10 total | Semantic 0–5 + contextual 0–5, parsed as a 0–10 total; evaluator emits a new 0–10 score | Separate semantic and contextual 0–5 tool arguments; the service adds them |
| Single-item result | String wrapped by the route as `{ "response": ... }` | Object with final answer and scores | Object with final answer and scores |
| Batch item | String | Object | Object |
| Persistence | CSV row with original/final scores, suggestions, revised answer, and reported cost | CSV row with original/final scores, suggestions, revised/final answer | CSV row with separate review state, decision, justification, and revision count |

## `user_reviewer`

The reviewer evaluates semantic and contextual quality, emits tagged scores, and suggests changes when the combined score is 7 or lower. An automated `UserProxyAgent` acts as the reviser: it reads those suggestions, writes a tagged revised answer, and sends the result back for another review. The reviewer permits two consecutive automatic replies and the proxy permits three, so this is a bounded conversation rather than an unbounded refinement loop.

The service parses the first two `<total_score>` values from chat history as the previous and new scores. It uses the first tagged `<revised_answer>` as the returned text; if none exists, it keeps the original answer. Missing final scores are written and exposed internally as `"-"`, while the HTTP route returns only the final string under `response`.

## `group_chat`

The reviewer, rewriter, and evaluator participate in a round-robin `GroupChat`. The reviewer emits semantic, contextual, and total scores plus suggestions. The rewriter produces a candidate from the question data and suggestions. The evaluator chooses between the original and revised answers and emits `<final_answer>` and `<new_score>` tags.

The manager stops when it sees the unanswerable sentinel or a reviewer total above 7. After the chat, the service scans up to three messages in reverse order for tagged values. It keeps the original answer when the original score is above 7. It converts a new score of 7 or lower to an internal `DO_NOT_ANSWER` result, then exposes that result as `"-"`. A missing new score is also represented as `"-"`.

Because both cases use the same sentinel, clients cannot distinguish “no revised score was produced” from the string form without considering the other fields, and a `final_answer` of `"-"` represents a rejected answer.

## `swarm`

The swarm routes explicitly between agents with AG2 tool return values:

1. The semantic reviewer registers a 0–5 score and hands off to the contextual reviewer.
2. The contextual reviewer registers a 0–5 score and sums the pair.
3. An original total above 8 terminates the workflow. Otherwise, the suggester and rewriter produce a candidate.
4. The revised candidate is reviewed again, then the decider chooses `ANSWER_REVISED`, `REWRITE`, or `DO_NOT_ANSWER`.
5. `REWRITE` promotes the revised answer and scores to the next round's original state before returning to the rewriter. The group chat is capped at 30 rounds.

The decider prompt says that after two or more revisions an unacceptable answer must become `DO_NOT_ANSWER`; that limit is prompt guidance. The tool code does not independently validate the decision or enforce the revision count. Similarly, score types and decision values depend on model tool calls.

The service applies a further post-processing rule: a new total of 7 or lower becomes `DO_NOT_ANSWER`, while an original total above 7 restores the request's original answer. This latter threshold differs from the contextual tool's early-termination threshold of greater than 8. The service exposes `DO_NOT_ANSWER` as `"-"` and uses `"-"` for a missing new score.

Two decision checks in `swarm/services/revision_service.py` compare strings with Python's `is` operator. Identity is not a reliable test of string value, so those post-processing branches may not consistently recognize `REWRITE` or `DO_NOT_ANSWER`.

## Shared constraints

All three services serialize the input into the agent prompt and derive the language as Portuguese only when `locale` is exactly `pt`; otherwise they label it Spanish. They append to a working-directory-relative `results.csv` without locking. Agent instances are created once at import time. The group-chat service reads the shared manager's chat history, while the swarm mutates one module-level `ContextVariables` object for all requests. These choices make the examples unsuitable for concurrent request processing without further isolation and can allow earlier chat state to affect later extraction.

Agent output is parsed from tags or accepted through model-generated tool arguments. Malformed or unexpected output can therefore produce missing scores or fallback values. The route handlers convert any exception into an HTTP 500 response containing the exception text.
