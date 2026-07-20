# Observed failure modes — Qwen2.5-1.5B-Instruct (CPU, greedy)

Empirical evidence behind each family's `failure_mode` field. Every claim below
links to a full transcript in `analysis/transcripts/`. Model: a deliberately
small local model (the point is the *process* — swap the backend and these
analyses regenerate; see README "per-model calibration").

## 1. Goal completion blindness → retry spiral with collateral damage

**Transcript:** `channel_setup.L1.0002.*.json` · **Task:** "Create a new public channel named #growth-ideas." · **Verdict:** FAIL — collateral channels `growth-ideas-new`, `growth-ideas-new2`

The task was *complete at step 2*. Everything after is self-inflicted:

1. Step 1: passes `"#growth-ideas"` with the `#` prefix → API rejects the name
   (the entity-format slip the family declares).
2. Step 2: retries without `#` → **success. The task is done.**
3. Steps 3–7: hallucinated objectives nobody asked for — invite `@everyone`
   (not a user), welcome message to `#general` (doesn't exist).
4. Steps 8–9: retries the original creation, gets "already exists" — and
   instead of reading that as *done*, treats it as an obstacle...
5. Steps 10, 18: ...creating `growth-ideas-new` and `growth-ideas-new2`.

Curriculum consequence: graders must check *absence of extras*, not just
presence of the target — a target-presence-only grader scores this episode 1.
This is exactly what the `extra_channel` mutant enforces for every instance of
the family.

## 2. Dropped step + overconfident self-report

**Transcript:** `channel_setup.L2.0002.*.json` · **Task:** create `#growth-ideas` + set header + add `@sofia` · **Verdict:** FAIL — `user 'sofia' is not a member`

Five clean tool calls: create → (gratuitous pinned-check) → set header →
`finish`. The third required action — inviting @sofia — is silently dropped,
and the finish summary asserts success ("created with proper header"). The
self-report is not evidence; only state verification catches this.

Curriculum consequence: L2 chains discriminate between models precisely on
step-tracking; the `wrong_member` and `missing_header` mutants pin the grader
to each individual step of the chain.

## 3. Distractor capture + asking instead of acting + degenerate spam loop

**Transcript:** `thread_reply.L3.0001.*.json` · **Task:** find the unanswered travel-budget question in ~q3-planning, reply in its thread with the approved figure (pinned in ~finance) · **Verdict:** FAIL — collateral posts in ['finance', 'q3-planning']

Three distinct failures in one episode:

1. **Distractor capture.** Search returns a seeded near-miss ("If we hadn't
   cut scope, travel budget would be around $45,000") and the model adopts
   $45,000 — a distractor — never checking the pinned post in ~finance
   (`get_pinned_messages` is never called).
2. **Asks instead of answers.** Its thread reply is a *question* ("Can anyone
   confirm...?") — it delegates the task back to the humans.
3. **Degenerate loop.** Steps 2–19: the same message pasted as new channel
   posts 18 times until the tool-call budget runs out — textbook small-model
   repetition under long context.

Curriculum consequence: this single episode motivates the `distractor_value`
mutant (near-miss figures must be rejected), the collateral snapshot-diff
check (the spam), and a difficulty knob (`n_distractors`) that directly
controls how strongly the family exercises distractor discrimination.

## 4. Belt-and-suspenders double posting

**Transcript:** `thread_reply.L3.0002.*.json` · **Verdict:** FAIL — collateral post in ['product']

The near-success case, and the most instructive: the model finds the correct
figure ($97,000), replies in a thread with it — and then *also* posts the same
answer as a fresh channel message "to be safe". Four efficient calls, right
value, and still a failed episode: duplicating an answer across surfaces is
noise pollution in a real workspace. Without the snapshot-diff collateral
check this scores 1, and RL would learn that spraying answers is free. This is
the false positive class the `correct_plus_spam` mutant guards against — the
episode is that mutant, produced organically by a real model.

## 5. Exploration without commitment (Qwen2.5-32B, L4)

**Transcripts:** `rollup_reply.L4.{0001,0003,0005}.ollama_qwen2.5_32b.json` · **Verdict:** FAIL — no reply in thread, 20/20 calls burned

The failure mode of the *strong* model, absent at every lower level, appearing
exactly when the curriculum entered its gradient band (L4 pass rate 40%).
Identical shape in all three failures:

1. **Hallucinated entity IDs.** Every episode opens with `get_thread` on an
   invented, descriptively-named id (`"sofia_q3_figure_request"`,
   `"diana_q_post_id"`, `"ingrids_question_post_id"`) — the model *narrates*
   the reference it wishes it had instead of discovering the real one. It
   recovers from the error, but the call is burned.
2. **Prompt-format leakage.** `read_channel("~q3-planning")` — the `~` sigil
   from the prompt's channel references is passed straight into the tool call,
   errors, and forces a `list_channels` round-trip. (Same class as the 1.5B's
   `#growth-ideas`, surviving 20× the parameters.)
3. **Exploration without commitment.** The remaining 15+ calls are all
   read/search/pinned — re-reading the same channels and pages, issuing
   near-duplicate searches — and the episode ends mid-exploration with zero
   actions taken. The model never satisfies its own completeness bar for the
   aggregation ("do I have ALL the components?") and exhausts the budget
   verifying instead of acting. The 2 passing episodes committed at 11 and 15
   calls; the margin to the 20-call cap is thin.

Curriculum consequences: (a) the max-tool-call budget is itself a difficulty
knob — tightening it converts near-passes into failures and directly pressures
the explore/act tradeoff RL should teach; (b) aggregation tasks with
uncertain component counts induce this failure class in strong models —
variants that state vs. withhold the component count ("these three budgets"
vs "all relevant budgets") should bracket it; (c) hallucinated-id calls are
cheap to detect mechanically (id not seen in any prior tool result) and could
feed a per-episode diagnostic signal alongside the binary reward.

## Cross-cutting observations

- Tool errors are recoverable for the model (it fixed `#name` → `name`), but
  each recovery consumes budget and raises the chance of goal drift — evidence
  that max-tool-call limits are themselves a difficulty knob.
- Self-reports ("finish" summaries) correlate weakly with success (2/2 of the
  false completions here carried confident success summaries). Final-state
  grading, never self-report grading.
- The model that *passed* L1.0001 and L2.0001 solved them in 4–5 calls; the
  failures burned 5–20. Call-efficiency is a useful secondary signal for
  calibration even under binary reward.
