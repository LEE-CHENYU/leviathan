# Implementation Compromises & Deferred Work

Date: 2026-02-24
Status: Living document — update as items are resolved

This document tracks every simplification, shortcut, and deferred decision made during Phase 0 and Phase 1 implementation. Each item notes what the design docs prescribed, what we actually built, why, and when it should be revisited.

---

## Phase 0: WorldKernel Extraction

### ~~K1. In-process code execution (no real sandboxing)~~ RESOLVED

- **Resolved:** 2026-02-24. Added `SubprocessSandbox` in `kernel/subprocess_sandbox.py` with `EngineProxy` action proxy pattern. Agent code runs in a child process with timeout enforcement (default 5s). Actions recorded as JSON by the proxy and applied to the real engine via `WorldKernel.apply_intended_actions()`. 7 sandbox tests + 8 proxy tests + 3 apply tests + 1 E2E integration test.

### K2. Snapshot only captures minimal member stats

- **Design says:** `WorldSnapshot.members` is `List[Dict]` — implies rich per-member state.
- **What we built:** Each member dict only has `{id, vitality, cargo, land_num}`. Missing: relationships/memory, gene values, decision parameters, age, health, children, strategy history.
- **Risk:** Snapshots are incomplete for full state reconstruction or detailed agent observation.
- **When to fix:** When the API needs to serve richer agent state (Phase 2 SDK, or dashboard features). Extend `get_snapshot()` to collect more fields from `Member` objects.

### K3. Active mechanisms tracking is shallow

- **Design says:** `active_mechanisms` should list currently active world rules.
- **What we built:** Only pulls from `execution_history["rounds"][-1]["mechanism_modifications"]["executed"]` — only shows mechanisms executed in the *last round*, not a cumulative registry of all active mechanisms.
- **Risk:** Clients see an incomplete picture of world rules. Mechanisms from earlier rounds that are still active won't appear.
- **When to fix:** When mechanism governance becomes important (Phase 3). Needs a proper mechanism registry in the kernel tracking activation, deactivation, and rollback state.

### K4. settle_round only runs produce/consume

- **Design says:** Settlement should run the full round phase sequence (produce, consume, environment, etc.).
- **What we built:** `settle_round()` calls only `self._execution.produce()` and `self._execution.consume()`. Missing: `fight`, `trade`, `reproduce`, `environment` phase, contract settlement, physics enforcement.
- **Risk:** Round settlement in the kernel doesn't match the full simulation loop. The kernel's golden tests verify determinism for a limited phase set.
- **When to fix:** When the kernel needs to fully replace the manual simulation loop. Either: (a) call all phase methods in `settle_round`, or (b) integrate with the DAG graph engine's `execute()` which already handles phase ordering.

### K5. No round_metrics or judge_results populated

- **Design says:** `RoundReceipt` has `round_metrics: Dict[str, float]` and `judge_results: List[Dict]`.
- **What we built:** Both are always empty (`{}` and `[]`). The kernel has no judge integration and doesn't compute per-round metrics.
- **Risk:** Receipts lack observability data. Consumers of receipts see no performance, fairness, or governance data.
- **When to fix:** Phase 3 (judge integration) for `judge_results`. Metrics should be added when eval_metrics is wired into the kernel.

### K6. Idempotency cache is per-round only, in-memory

- **Design says:** Idempotency keys prevent duplicate execution.
- **What we built:** Cache resets on every `begin_round()`. No cross-round or cross-restart deduplication.
- **Risk:** If an action is retried after a round boundary or server restart, it will execute again.
- **When to fix:** Phase 2 (external write path). Needs persistent idempotency store (SQLite or similar) keyed by `(world_id, idempotency_key)`.

### K7. No receipt signing or oracle identity

- **Design says (06-united-leviathan.md Prep 2):** Each world should have an oracle signing identity. Receipts should be signed by the world oracle key.
- **What we built:** No signing. Receipts are plain data. No `world_public_key` or signature fields.
- **Risk:** Receipts cannot be cryptographically verified. Cross-world federation has no trust anchor.
- **When to fix:** Before federation (post-Phase 4). Add keypair generation on world creation, sign receipts with `oracle_signature` field.

### K8. No federation-ready fields in receipts

- **Design says (06-united-leviathan.md Prep 3):** Reserve fields: `origin_world_id`, `origin_receipt_hash`, `bridge_channel_id`, `bridge_seq`, `notary_signature`.
- **What we built:** None of these fields exist in `RoundReceipt`.
- **Risk:** Adding them later may require schema migration and client updates.
- **When to fix:** Before any cross-world feature. Can be added as `Optional` fields to `RoundReceipt` without breaking existing code.

### K9. No constitution object or constitution_hash in receipts

- **Design says (05-meta-edit.md):** Formalize a `Constitution` object with unamendable kernel clauses, amendable governance clauses, and open world rules. Include `constitution_hash` in every round receipt.
- **What we built:** No constitution concept. Receipts have no `constitution_hash`.
- **Risk:** No formal specification of what invariants the world guarantees. Makes trust/audit harder.
- **When to fix:** Phase 4 (moderator + multi-tenant) or when external agents need trust guarantees.

### K10. WorldKernel constructor requires save_path

- **Design says:** `WorldKernel.__init__(self, config: WorldConfig)` — clean single-argument constructor.
- **What we built:** `WorldKernel.__init__(self, config: WorldConfig, save_path: str)` — requires a filesystem path because `IslandExecution` needs one.
- **Risk:** Kernel isn't truly "no filesystem side effects" as the docstring claims. IslandExecution writes to disk.
- **When to fix:** When refactoring IslandExecution to support in-memory mode, or when the kernel fully encapsulates storage.

### ~~K11. No integration test comparing WorldKernel vs direct IslandExecution~~ RESOLVED

- **Resolved:** 2026-02-24. Added `TestKernelIslandEquivalence` with 3 tests: initial state equivalence, produce/consume equivalence, and multi-round (3 rounds) equivalence. All compare WorldKernel snapshot members against direct IslandExecution members field-by-field with pytest.approx for floats.

### K12. messages_sent is always empty in ActionResult

- **Design says:** `ActionResult.messages_sent: List[Tuple[int, str]]` — captures messages agents send during actions.
- **What we built:** Always `[]`. The sandbox doesn't capture broadcast messages from agent code.
- **Risk:** Inter-agent communication is invisible to the kernel.
- **When to fix:** When message passing becomes a kernel concern (Phase 2 or federation).

---

## Phase 1: Read API

### A1. Single world only — no /v1/worlds (plural)

- **Design says:** `GET /v1/worlds` lists all worlds.
- **What we built:** `GET /v1/world` (singular) serves one world. No multi-world support.
- **Upgrade path:** Add `/v1/worlds/{id}/...` routes alongside `/v1/world/...` in Phase 2. Singular routes become aliases.
- **When to fix:** Phase 2 or when multi-world is needed.

### A2. In-memory event log — no persistence

- **Design says:** "Storage: SQLite initially."
- **What we built:** Events stored in a Python list. Lost on restart.
- **Risk:** No historical data survives server restarts. No query capability beyond linear scan.
- **When to fix:** When persistence matters (production deployment or long-running sims). Add SQLite event store behind the same interface.

### A3. Polling only — no SSE streaming

- **Design says:** Event feed should support real-time observation.
- **What we built:** `GET /v1/world/events?since_round=N` polling endpoint only. No Server-Sent Events.
- **Risk:** Clients must poll repeatedly. Higher latency for real-time dashboards.
- **When to fix:** Phase 2 or when a real-time dashboard is needed. `sse-starlette` is already installed.

### A4. No event_seq, phase, payload_hash, or prev_event_hash in EventEnvelope

- **Design says (06-united-leviathan.md Prep 1):** Canonical event envelope should include `world_id`, monotonic `event_seq`, `round_id`, `phase`, `type`, `payload_hash`, `prev_event_hash`.
- **What we built:** `EventEnvelope` has: `event_id`, `event_type`, `round_id`, `timestamp`, `payload`. Missing: `world_id`, `phase`, `payload_hash`, `prev_event_hash`.
- **Risk:** Events aren't self-contained for federation verification. No hash chain for tamper detection.
- **When to fix:** Before federation. Add missing fields as `Optional` to `EventEnvelope`.

### A5. No CORS configuration

- **What we built:** No CORS middleware. Browser-based dashboards will be blocked.
- **When to fix:** When a web frontend needs to call the API. One-line FastAPI middleware addition.

### ~~A6. No rate limiting or request validation~~ RESOLVED

- **Resolved:** 2026-02-24. Added `api/auth.py` with `APIKeyAuth` dependency (X-API-Key header or query param, 401/403 responses) and `RateLimiterMiddleware` (per-IP token bucket, 429 on exhaustion). Auth is opt-in per route via `Depends(get_auth)` — Phase 1 read routes stay open, Phase 2 write routes will require it. 8 tests added.

### A7. Event log grows unbounded in memory

- **What we built:** Events append to a list forever. No eviction, no pagination limit enforcement.
- **Risk:** Memory grows linearly with rounds. At ~1KB per event and 1 round/2s, that's ~1.7GB/day.
- **When to fix:** Before long-running deployments. Add a max event count with FIFO eviction, or switch to SQLite (A2).

### A8. No OpenAPI schema customization or versioning

- **What we built:** FastAPI auto-generates OpenAPI docs at `/docs`. No version pinning, no schema export.
- **Risk:** API consumers have no stable schema contract.
- **When to fix:** Before external SDK development (Phase 2). Pin OpenAPI spec version, export to file, add to CI.

### ~~A9. Simulation loop is not integrated with real agent LLM calls~~ RESOLVED

- **Resolved:** 2026-02-24. Phase 2 adds submission window: `begin_round → open_submissions → sleep(pace) → close_submissions → execute_actions → settle_round`. External agents submit via `POST /v1/world/actions` during the window. Actions run through `SubprocessSandbox` and applied via `WorldKernel.apply_intended_actions()`.

### A10. No thread safety for concurrent API reads during simulation writes

- **What we built:** Background sim thread mutates kernel state while API threads read it. No locking.
- **Risk:** Race conditions where a snapshot read overlaps with a settle_round write, producing inconsistent data.
- **When to fix:** Before production. Add a `threading.RLock` around kernel state access, or move to async-only architecture with explicit phase boundaries.

---

## Summary by Priority

### Must fix before Phase 2 (external agents)
- ~~**K1** (sandbox security)~~ RESOLVED — SubprocessSandbox with action proxy pattern
- ~~**K11** (integration test)~~ RESOLVED
- ~~**A6** (rate limiting + auth)~~ RESOLVED

### Should fix before production
- **K6** (persistent idempotency) — cross-restart deduplication, nice-to-have not blocker
- **A10** (thread safety) — Python GIL makes most ops safe enough for dev; add locking before production

### Should fix before Phase 3 (governance)
- **K3** (mechanism registry) — governance needs full mechanism tracking
- **K4** (full settlement) — kernel settlement must match the real sim loop
- **K5** (metrics + judge) — governance needs observability

### Should fix before Phase 4 (multi-tenant / federation)
- **K7** (receipt signing) — trust requires cryptographic verification
- **K8** (federation fields) — schema must be ready before clients depend on current shape
- **K9** (constitution) — trust model needs formal specification
- **A4** (event envelope enrichment) — federation needs self-contained events

### Fix when convenient (low urgency)
- **K2** (richer snapshots) — extend when needed
- **K10** (save_path coupling) — cosmetic, works fine
- **K12** (messages_sent) — wire when messaging matters
- **A1** (multi-world) — add alongside, no breaking change
- **A2** (SQLite) — add when persistence matters
- **A3** (SSE) — add when real-time matters
- **A5** (CORS) — one-line fix when frontend exists
- **A7** (memory eviction) — add before long-running deployments
- **A8** (OpenAPI pinning) — add before SDK release
- ~~**A9** (headless sim)~~ RESOLVED — Phase 2 write path adds submission window
