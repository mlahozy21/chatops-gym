# chatops-gym

**A verifiable-task curriculum generator for RL training of software agents**, built on a real Slack-like application (Mattermost) running in a single Docker container.

This is not a benchmark. Benchmarks (τ-bench, AgentBench, WebArena) are fixed, hand-curated sets built to *measure* models. This is a **curriculum generator** built to *train* them — and the three things a training curriculum needs that a benchmark doesn't are exactly what this repo is about:

1. **Procedural generation with consistency by construction** — every task instance derives its prompt, world state, deterministic grader *and* reference (oracle) solution from the same sampled template parameters. Thousands of instances, zero hand-drift between what the task asks and what the grader checks.
2. **Per-model difficulty calibration** — a structural L1–L4 taxonomy as a prior, re-measured empirically against any model backend in hours, because the useful RL gradient band (~10–60% pass rate) is a property of the *model*, not the task.
3. **Graders that are themselves validated, in CI** — because at generation scale nobody reviews graders by hand, and a buggy grader poisons the reward signal.

## The validation harness (the interesting part)

Every task must pass this contract before it ships — it runs in CI on every PR:

| Check | Required verdict | What a failure means |
|---|---|---|
| **Oracle through agent tools** | 1 | The reference solution, executed through the *same 13 tools, same user, same permissions* as the evaluated agent, must solve the task. This proves "solvable by the agent", not merely "solvable by the API" — a task requiring a capability the tools don't expose fails here. |
| **Every mutant** | 0 | *Mutation testing*: the oracle solution is systematically perturbed into plausible near-misses — reply in the channel instead of the thread, use a seeded distractor figure, negate the correct answer ("definitely not $48,500"), do everything right *plus* spam another channel, hedge between two values, do nothing. The grader must reject every one. |
| Null agent | 0 | Grader must not be satisfied by the initial world state. |
| Random agent (n seeds) | 0 | Grader must not be satisfied by arbitrary world mutations. |

Null/random are standard smoke tests — deliberately weak. **Mutation testing is the centerpiece**: the false positives that poison RL come from *plausible near-misses*, because a model mid-training produces almost-correct behavior, not noise. Mutants derive from the same template parameters as the task, so each family gets them nearly for free.

The harness earns its keep in practice: its first run against a live server caught a real grader bug (Mattermost attributes system join-messages to the acting user; the collateral-damage check counted them as agent posts and failed the oracle).

```
$ python -m validation.validate_curriculum
=== thread_reply.L3.0001 (L3) ===
  [ok] oracle_through_tools             expected=1 got=1 (ok)
  [ok] mutant:channel_instead_of_thread expected=0 got=0 (no agent reply in the target thread)
  [ok] mutant:distractor_value          expected=0 got=0 (value 45000 not found in reply)
  [ok] mutant:negated_value             expected=0 got=0 (value 45000 only appears negated)
  [ok] mutant:correct_plus_spam         expected=0 got=0 (collateral posts in: ['random'])
  [ok] mutant:read_only                 expected=0 got=0 (no agent reply in the target thread)
  [ok] mutant:hedged_answer             expected=0 got=0 (ambiguous reply: also asserts [54000.0])
  [ok] null_agent                       expected=0 got=0 ...
curriculum is valid: every task solvable via agent tools,
every grader rejects all near-miss mutants.
```

## Anatomy of a task

Tasks are self-contained JSON (schema in `tasks/schema.json`): a deterministic world (seed + overlays), a natural-language prompt, a parameterized grader, and a mandatory oracle solution.

```jsonc
{
  "task_id": "thread_reply.L3.0042",
  "difficulty": "L3",
  "failure_mode": "replies in channel instead of thread; picks distractor figure",
  "prompt": "Someone in ~q3-planning asked about the marketing budget and never got
             an answer. Find that message and reply in its thread with the correct
             figure. The approved figure is recorded somewhere in ~finance.",
  "setup": {
    "world_seed": 42,          // -> 12 users, 8 channels, ~60 background messages
    "overlays": [              // the task's own deltas, anchored by markers
      {"op": "post_message", "channel": "q3-planning", "user": "diana",
       "text": "Does anyone know the final marketing budget for Q3?",
       "marker": "root_question"},
      {"op": "pin_message", "channel": "finance",
       "text": "Q3 marketing budget approved: $48,500", "marker": "answer_pin"}
    ]
  },
  "grader": {"module": "graders.families.thread_reply",
             "params": {"root_marker": "root_question",
                        "answer": {"type": "money", "value": 48500}}},
  "oracle_solution": [ /* tool calls; references re-found THROUGH the tools */ ]
}
```

Design decisions worth knowing:

- **Stable keys, never IDs.** Mattermost entity IDs are not stable across resets. Overlays register `marker -> entity` per episode; graders (trusted infrastructure) resolve markers directly. The **oracle may not**: it re-finds entities through the agent's own tools (`post_in_channel` / `pinned_post` resolvers), and the marker registry is only used to *cross-check* that it found the right one — if not, validation fails loudly, exposing a weakly-anchored task.
- **Typed answers, not substrings.** `"must_contain": "48,500"` passes on *"definitely not $48,500"*. Graders declare typed expected values (`money`, `text`, `number`) matched with normalization ($48,500 / 48500 / 48.5k), a negation guard, and ambiguity rejection (asserting two values fails). Residual limitation: this is shallow NLP over free text, documented below.
- **Final state, not trajectory** (as in τ-bench), hardened with a **snapshot-diff**: the env snapshots the world at reset; graders diff against it, so collateral damage (spam, stray channels, header vandalism) is caught generically without enumerating what the agent shouldn't touch.
- **Episode isolation by namespacing.** Every episode is a fresh Mattermost *team*, seeded deterministically. This is also the concurrency answer: N parallel rollouts = N teams (+ a container pool for coarse parallelism); markers and graders are already team-scoped.

## Difficulty: structural prior, per-model truth

| Level | Structural definition | Example |
|---|---|---|
| L1 | 1 action, entity named explicitly | create channel `#q3-budget` |
| L2 | 2–4 chained actions, no retrieval | create + set header + invite |
| L3 | retrieval across channels w/ distractors + threaded action | the task above |
| L4 | multi-channel workflow w/ aggregation or ambiguity | triage all unanswered questions |

The taxonomy is a **prior**. Difficulty in RLVR is relative to the model: a 7B's 10–60% gradient band is not a frontier model's. The claim this repo makes is about *process*: `calibration/report.py` re-measures the full curve against any backend and flags tasks outside that model's useful band (`SATURATED` / `TOO HARD`), and templates expose **continuous difficulty knobs** (number of distractors, history depth the needle is buried in) so the curriculum regenerates *upward* when a model saturates it — the core argument for a generator over a fixed benchmark.

```bash
python -m scripts.run_eval --backend ollama:qwen2.5:7b
python -m calibration.report results/ollama_qwen2.5_7b.jsonl
```

No local GPU? `notebooks/colab_ollama_server.ipynb` serves Qwen2.5-7B from a
free Colab GPU behind a Cloudflare tunnel; point the local harness at it with
`OLLAMA_URL=https://<tunnel>.trycloudflare.com` — no code changes. Any
OpenAI-compatible endpoint also works via `OPENAI_BASE_URL` (Groq, vLLM, ...).

Failure transcripts are first-class outputs (`analysis/transcripts/`): each family's `failure_mode` field cites observed agent behavior, not intuition. For frontier-model failure modes a local 7B can't exhibit, we lean on published evidence (τ-bench's consistency results; TheAgentCompany's long-horizon failures).

### First empirical slice (Qwen2.5-1.5B-Instruct, CPU, greedy, n=6)

```
level       n   pass-rate
L1          2        50%      channel_setup     [in band]
L2          2        50%      channel_setup     [in band]
L3          2         0%      thread_reply      [TOO HARD for this model -> lower knobs]
```

Small-n vertical slice, but the process already works end-to-end: the curve is
monotone, the report flags L3 as out of the 1.5B's gradient band, and the four
failures produced four annotated archetypes (`analysis/FAILURE_MODES.md`):
goal-completion blindness with a retry spiral, a silently dropped step behind
a confident success summary, distractor capture plus a degenerate spam loop,
and a belt-and-suspenders double post that organically reproduced the
`correct_plus_spam` mutant. Two of the six episodes were false completions
whose `finish` summaries claimed success — self-reports are not evidence;
final-state grading is.

### Same curriculum, bigger model: saturation observed (Qwen2.5-32B via Ollama)

```
              1.5B (CPU)         32B (GPU)
L1               50%               100%   (2 calls/solve)
L2               50%               100%   (4 calls/solve)
L3                0%               100%   (4 calls/solve)
```

The 32B saturates the entire base curriculum with near-optimal call counts,
and the calibration report flags every family SATURATED for that model — the
same tasks sit *below* the 32B's gradient band and *above* the 1.5B's at L3.
Difficulty is a property of the (task, model) pair, measured, not asserted.

**And the honest second finding: quantitative knobs weren't enough.** A
`--hard` batch (6 distractors, 150-deep history, validated 15/15 by the
harness) was *also* saturated by the 32B, 15/15 at 2–5 calls. Root cause,
visible in the episode traces: the L3 answer is *pinned*, so
`get_pinned_messages` gives a strong model an O(1) retrieval path — no
distractor count competes with a pin. When knobs stop moving a model, the
difficulty gap is structural, and the curriculum answers with a structurally
harder family: `rollup_reply` (L4) removes the shortcut entirely — no pins,
final figures mixed with superseded drafts ("early proposal — NOT final"),
and the asked-for number existing *nowhere* in the workspace: the agent must
find N component figures across N channels, discriminate final from draft,
sum them, and post exactly one threaded reply. Its mutation suite covers the
new false-positive classes: partial sums, sums built from a draft figure, and
single-component answers.

**The loop closes with L4 in band.** `rollup_reply` puts the 32B at **40%**
— inside the useful gradient zone — and its three failures exhibit a failure
mode the model never showed at lower levels: *exploration without commitment*
(20/20 calls burned re-reading channels, zero actions taken), plus
hallucinated post-ids and prompt-sigil leakage (`~channel` passed verbatim to
tools). Full annotated traces in `analysis/FAILURE_MODES.md` §5.

```
Final curve — ollama:qwen2.5:32b (n=5/level)
L1  100%  ########################################   saturated
L2  100%  ########################################   saturated
L3  100%  ########################################   saturated (pin shortcut)
L4   40%  ################                           IN BAND (10-60%)
```

That saturation → diagnose → structural response → back-in-band loop, with
every step validated in CI and measured against a real model, is the
curriculum engineering workflow this repo exists to demonstrate.

## Quickstart

```bash
docker compose -f docker/docker-compose.yml up -d   # the environment (a commodity)
pip install -r requirements.txt

make test        # unit tests, no server needed
make generate    # templates -> tasks/instances/*.json
make validate    # the harness: oracle=1, mutants=0, null=0, random=0
make eval        # real model via Ollama (or: --backend hf:Qwen/Qwen2.5-0.5B-Instruct,
                 #   a CPU-only fallback; or anthropic:/openai: with an API key)
make report      # per-model difficulty calibration
```

## Adding a task family (~30 lines + mutants)

1. `tasks/templates/<family>.py` — a `TaskTemplate` whose `generate(seed, **knobs)` derives prompt, overlays, grader params and oracle from the same sampled parameters.
2. `graders/families/<family>.py` — a `Grader` returning `Verdict(passed, reason)`; use `env.diff_since_reset()` for collateral checks and `graders.matchers` for typed answers.
3. `validation/mutations.py` — register ≥3 named near-miss mutants. **The harness refuses to validate a family without mutants.**
4. `make generate && make validate`.

## What this grading model can't verify (known limits)

Ephemeral actions (deleted before grading), pure-read tasks that change no state, "don't do X" prohibitions that leave no trace in the diff, destroy-then-restore trajectories that land on a correct final state, and post-grading edits. Some are mitigable (grading over the audit log instead of final state); all are declared rather than papered over. The typed matcher is shallow NLP: it kills the cheap false-positive classes (negation, ambiguity, format variance), not adversarial paraphrase.

## Repo map

```
tasks/         templates (the generators) + instances (the curriculum) + schema
graders/       typed matchers + per-family graders (final-state + snapshot-diff)
validation/    mutation testing + the ship/no-ship harness (runs in CI)
gym/           env (episode teams, markers, snapshot), 13 tools, runner, agents
calibration/   per-model difficulty reports
analysis/      annotated failure transcripts — the evidence behind failure_mode
docker/        one docker compose up
```

## License

Released under the MIT License — see `LICENSE`.
