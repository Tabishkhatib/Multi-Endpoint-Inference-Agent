# Multi-Endpoint Inference Routing Agent

An agent that routes queries across three local Ollama endpoints, detects
degradation or failure **mid-stream**, and decides in real time whether to
switch endpoints, wait, or accept partial output  logging the reasoning
behind every decision.

## What it does

1. Classifies an incoming query (code / reasoning / simple).
2. Routes it to the endpoint best suited for that category, based on that
   endpoint's live health and historical success rate — not a fixed order.
3. Streams the response token by token, timing the gap between tokens.
4. Compares each gap against that endpoint's own rolling baseline speed.
5. If degradation looks severe (or the connection dies outright), hands off
   to a Decision Engine that picks one of three outcomes:
   - **SWITCH**  abandon this endpoint, restart the query on a healthy
     backup, carrying forward the text already generated
   - **WAIT**  grant one grace period rather than throw away work that's
     nearly finished
   - **ACCEPT_PARTIAL** stop and return what's been generated so far,
     honestly flagged as incomplete, when switching would cost more than
     it recovers
6. Every endpoint's performance is persisted between runs, so the agent's
   routing decisions get smarter over time.

## Architecture

| File | Responsibility |
|---|---|
| `ollama_client.py` | Streams tokens from one endpoint, timestamps each one, distinguishes a real dead connection from a slow one |
| `endpoint_stats.py` | Tracks each endpoint's rolling baseline speed and success rate; handles the calibration-to-baseline transition |
| `classifier.py` | Keyword-based query categorization (code / reasoning / simple)  used as the automatic fallback |
| `semantic_classifier.py` | Primary classifier: real embedding-based semantic similarity (nomic-embed-text), falls back to `classifier.py` if the embedding call fails or times out |
| `router.py` | Picks the starting endpoint using classification + live health + success rate |
| `decision_engine.py` | The WAIT / SWITCH / ACCEPT_PARTIAL logic, given a degradation event |
| `agent.py` | Orchestrates all of the above into one end-to-end run |
| `fault_injector.py` | Kills or suspends a real Ollama process on demand, for reliable demo purposes |

## Endpoints

Three separate Ollama server processes, each on its own port, each serving
a different model:

| Port | Model | Role |
|---|---|---|
| 11437 | `llama3.2:1b` | Fast / simple queries |
| 11435 | `qwen2.5:3b` | Reasoning-heavy queries |
| 11436 | `qwen2.5-coder:1.5b` | Code queries |

They're separate **processes**, not just separate model names on one
server, so that one can genuinely be killed or throttled without affecting
the other two — this is what makes the fault injection honest rather than
simulated inside the agent's own code.

## How to run

1. Install [Ollama](https://ollama.com/download) and pull the four models:
   ```
   ollama pull llama3.2:1b
   ollama pull qwen2.5:3b
   ollama pull qwen2.5-coder:1.5b
   ollama pull nomic-embed-text
   ```
2. Start three independent server instances, one per terminal:
   ```
   $env:OLLAMA_HOST="127.0.0.1:11437"; ollama serve
   $env:OLLAMA_HOST="127.0.0.1:11435"; ollama serve
   $env:OLLAMA_HOST="127.0.0.1:11436"; ollama serve
   ```
3. Install Python dependencies:
   ```
   python -m pip install requests psutil
   ```
4. Warm up the embedding classifier once (avoids a ~15-20s cold-start delay
   on the first real query):
   ```
   python -c "from semantic_classifier import warm_up; warm_up()"
   ```
5. Run a query:
   ```
   python agent.py
   ```
6. To see it switch endpoints mid-stream, in a second terminal, while
   `agent.py` is streaming:
   ```
   python fault_injector.py kill 11437      # simulates a hard connection drop
   python fault_injector.py suspend 11437   # simulates a stall (use with fault_injector.py resume 11437 to undo)
   ```

## Key design decisions

**Classifier uses real semantic embeddings, with a keyword fallback.**
The primary classifier (`semantic_classifier.py`) embeds the incoming query
using Ollama's `nomic-embed-text` model and compares it via cosine
similarity against a handful of example queries per category. This catches
queries that mean the same thing as an example without sharing its exact
words  e.g. "there's a bug in my script" correctly classifies as a code
query even though it shares no literal keywords with terms like "function"
or "error".

An earlier prototype used bag-of-words vectors (hand-rolled, no pretrained
model) instead of real embeddings. That was rejected: bag-of-words only
catches word overlap, not meaning, so it wouldn't reliably separate
"reasoning" from "simple" queries, which differ in structure more than
vocabulary.

The embedding call is itself just another endpoint request, so it gets the
same degrade-gracefully treatment as everything else in this project: a
short timeout (5s) on real per-query calls, and an automatic fallback to
the keyword classifier (`classifier.py`) if the embedding call times out
or errors. The embedding model has a one-time cold-start cost (~15-20s on
first load)  a `warm_up()` function pays this cost upfront, before serving
real queries, so it doesn't land on the first real request.

**Degradation detection is calibration-then-baseline.** A fresh endpoint
has no history to compare against, so the first ~12 tokens use a fixed
fallback threshold. After that, degradation is judged relative to that
specific endpoint's own rolling average gap (5–10x baseline = mild,
20–30x = severe), because different model sizes have genuinely different
normal speeds a single global threshold would misjudge smaller/faster
models as "fine" when they're actually stalling.

**The switch/wait boundary is a 40% progress ratio.** Expected response
length is a rough per-category estimate (75/300/200 tokens for
simple/reasoning/code). Below 40% of that estimate, restarting elsewhere
costs less than it saves; above 40%, the sunk cost of tokens already
generated outweighs the benefit of a fresh start. This is a judgment call,
not a derived constant, and it's tunable.

**Fault injection kills/suspends the real OS process**, rather than faking
a delay inside the agent's own code. This means the agent is genuinely
detecting real connection drops and real stalls, not scripted ones.

## Known limitations (honest account)

- **Expected response length is a rough heuristic per category, not
  measured.** A "simple" query that happens to produce a long answer (a
  recipe, for example) will look artificially close to "done" early on,
  biasing the Decision Engine toward WAIT even when the response is still
  fairly short. A better version would learn typical response length per
  category from actual observed data, not a fixed guess.
- **The classifier can still misclassify some queries**  the embedding
  version is more accurate than the keyword fallback, but it's comparing
  against only 5 example queries per category, not a trained model. The
  cost of misclassification is a suboptimal initial routing choice, not a
  failed response.
- **The embedding model's cold-start cost (~15-20s) has to be paid
  explicitly via `warm_up()`** before serving real queries if that step
  is skipped, the first real query pays that cost instead, which could be
  mistaken for endpoint degradation if it happens mid-demo.
- **All three Ollama instances share one GPU**, so under real concurrent
  load they could contend for VRAM in a way that a genuinely distributed
  deployment wouldn't. This wasn't a factor in testing (queries run
  sequentially), but it's a real difference from a production setup.
- **Progress-ratio calibration is per-category, not per-query.** It
  doesn't account for the fact that queries within a category can vary a
  lot in actual length.

## What I'd do next with more time

- Grow the example set per category beyond 5 queries each, and consider
  learning prototypes from real logged queries over time instead of a
  fixed hand-written list.
- Learn expected response length per category from real observed data
  instead of a fixed guess.
- Test and tune the WAIT path more thoroughly it's implemented and has
  fired correctly in testing, but deserves more scenario coverage than a
  single grace period.
- Add concurrent/racing requests to multiple endpoints for latency-critical
  queries, rather than only reacting after committing to one endpoint.
