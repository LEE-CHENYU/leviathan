# Implementation Compromises & Deferred Work

Date: 2026-02-24
Status: Living document — update as items are resolved

This document tracks every simplification, shortcut, and deferred decision made during implementation. Each item notes what the design docs prescribed, what we actually built, why, and when it should be revisited.

---

## Phase 0: WorldKernel Extraction

### ~~K1. In-process code execution (no real sandboxing)~~ RESOLVED

- **Resolved:** 2026-02-24. Added `SubprocessSandbox` in `kernel/subprocess_sandbox.py` with `EngineProxy` action proxy pattern. Agent code runs in a child process with timeout enforcement (default 5s). Actions recorded as JSON by the proxy and applied to the real engine via `WorldKernel.apply_intended_actions()`. 7 sandbox tests + 8 proxy tests + 3 apply tests + 1 E2E integration test.

### ~~K2. Snapshot only captures minimal member stats~~ RESOLVED

- **Resolved:** 2026-02-24. Extended `get_snapshot()` to include `age`, `productivity`, `strength`, `consumption`, and `overall_productivity` per member. Still excludes relationships/memory, gene values, decision parameters, and strategy history (not needed for external agent observation).

### ~~K3. Active mechanisms tracking is shallow~~ RESOLVED

- **Resolved:** 2026-02-24. Added `MechanismRegistry` in `kernel/mechanism_registry.py` with full lifecycle tracking (submitted -> approved/rejected -> active). Registry provides `get_pending()`, `get_active()`, `get_all()`, `get_by_id()`. One-proposal-per-agent-per-round enforcement. API endpoints at `/v1/world/mechanisms/`. 11 registry tests + 7 endpoint tests.

### ~~K4. settle_round only runs produce/consume~~ RESOLVED

- **Resolved:** 2026-02-24. Added `KernelDAGRunner` in `kernel/dag_runner.py` that executes infrastructure phases in topological order (contracts -> produce -> consume -> environment). `settle_round()` now delegates to the DAG runner instead of manually calling `produce()` + `consume()`. 4 DAG runner tests + 4 settlement tests verify phase ordering and determinism.

### ~~K5. No round_metrics or judge_results populated~~ RESOLVED

- **Resolved:** 2026-02-24. `round_metrics` computed by `compute_round_metrics()` in `kernel/round_metrics.py` (total_vitality, gini_coefficient, trade_volume, conflict_count, mechanism_proposals, mechanism_approvals, population). `judge_results` populated from `JudgeAdapter` evaluations in the simulation loop. Both included in `RoundReceipt`. API endpoints at `/v1/world/metrics`, `/v1/world/metrics/history`, `/v1/world/judge/stats`. 8 metrics tests + 5 endpoint tests + 1 governance integration test.

### ~~K6. Idempotency cache is per-round only, in-memory~~ RESOLVED

- **Resolved:** 2026-02-24. Added `idempotency_keys` table to SQLite `Store`. `WorldKernel` persists action results to SQLite on every cache write and checks SQLite on cache miss. Old keys cleaned up on `begin_round()`. Cross-restart deduplication supported within the current round.

### ~~K7. No receipt signing or oracle identity~~ RESOLVED

- **Resolved:** 2026-02-24. Added `OracleIdentity` in `kernel/oracle.py` with Ed25519 keypair (deterministic from seed). Every `RoundReceipt` now includes `oracle_signature` (hex Ed25519 signature of canonical receipt JSON), `world_public_key` (oracle identity), and `constitution_hash`. 6 oracle tests + 6 kernel signing tests.

### ~~K8. No federation-ready fields in receipts~~ RESOLVED

- **Resolved:** 2026-02-24. Added `Optional` fields to `RoundReceipt`: `origin_world_id`, `origin_receipt_hash`, `bridge_channel_id`, `bridge_seq`, `notary_signature`. All `None` until federation is implemented. 4 receipt field tests.

### ~~K9. No constitution object or constitution_hash in receipts~~ RESOLVED

- **Resolved:** 2026-02-24. Added `Constitution` in `kernel/constitution.py` with 3-layer model (kernel immutable, governance amendable, world rules open). Loaded from `kernel/constitution_default.yaml`. `constitution_hash` (SHA-256) included in every `RoundReceipt`. 8 constitution tests.

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

### ~~A2. In-memory event log — no persistence~~ RESOLVED

- **Resolved:** 2026-02-24. Added `kernel/store.py` (SQLite with WAL mode) and `kernel/event_log.py`. Events persisted to `leviathan.db` in `LEVIATHAN_DATA_DIR`. `EventLog` class provides list-like interface (`append`, `__iter__`, `__len__`) so existing route handlers work unchanged. Fly.io volume mount added for production persistence.

### ~~A3. Polling only — no SSE streaming~~ RESOLVED

- **Resolved:** 2026-02-24. Added `GET /v1/world/events/stream` SSE endpoint using `sse-starlette`. Replays existing events matching the `since_round` filter, then streams new events in real-time as they are appended. Each SSE message includes `event` type, `id`, and JSON `data`.

### ~~A4. No event_seq, phase, payload_hash, or prev_event_hash in EventEnvelope~~ RESOLVED

- **Resolved:** 2026-02-24. Added `Optional` fields to `EventEnvelope`: `world_id`, `phase`, `payload_hash`, `prev_event_hash`. 2 enrichment tests.

### ~~A5. No CORS configuration~~ RESOLVED

- **Resolved:** 2026-02-24. Added `CORSMiddleware` to `api/app.py` with `allow_origins=["*"]`.

### ~~A6. No rate limiting or request validation~~ RESOLVED

- **Resolved:** 2026-02-24. Added `api/auth.py` with `APIKeyAuth` dependency (X-API-Key header or query param, 401/403 responses) and `RateLimiterMiddleware` (per-IP token bucket, 429 on exhaustion). Auth is opt-in per route via `Depends(get_auth)` — Phase 1 read routes stay open, Phase 2 write routes will require it. 8 tests added.

### ~~A7. Event log grows unbounded in memory~~ RESOLVED

- **Resolved:** 2026-02-24. Events now backed by SQLite (A2). In-memory cache still grows but is bounded by actual event count; SQLite handles the durable storage. For very long-running sims, the in-memory cache could be capped with LRU eviction, but SQLite resolves the data loss concern.

### A8. No OpenAPI schema customization or versioning

- **What we built:** FastAPI auto-generates OpenAPI docs at `/docs`. No version pinning, no schema export.
- **Risk:** API consumers have no stable schema contract.
- **When to fix:** Before external SDK development (Phase 2). Pin OpenAPI spec version, export to file, add to CI.

### ~~A9. Simulation loop is not integrated with real agent LLM calls~~ RESOLVED

- **Resolved:** 2026-02-24. Phase 2 adds submission window: `begin_round → open_submissions → sleep(pace) → close_submissions → execute_actions → settle_round`. External agents submit via `POST /v1/world/actions` during the window. Actions run through `SubprocessSandbox` and applied via `WorldKernel.apply_intended_actions()`.

### ~~A10. No thread safety for concurrent API reads during simulation writes~~ RESOLVED

- **Resolved:** 2026-02-24. Added `threading.RLock` to `WorldKernel`. All public methods (`get_snapshot`, `begin_round`, `accept_actions`, `accept_mechanisms`, `apply_intended_actions`, `settle_round`, `get_round_receipt`) acquire the lock. Reentrant lock allows `settle_round` to call `get_snapshot` internally.

---

## Summary by Priority

### Must fix before Phase 2 (external agents)
- ~~**K1** (sandbox security)~~ RESOLVED — SubprocessSandbox with action proxy pattern
- ~~**K11** (integration test)~~ RESOLVED
- ~~**A6** (rate limiting + auth)~~ RESOLVED

### ~~Should fix before production~~ ALL RESOLVED
- ~~**K6** (persistent idempotency)~~ RESOLVED — SQLite `idempotency_keys` table, cross-restart deduplication
- ~~**A10** (thread safety)~~ RESOLVED — `threading.RLock` on all public WorldKernel methods

### ~~Should fix before Phase 3 (governance)~~ ALL RESOLVED
- ~~**K3** (mechanism registry)~~ RESOLVED — MechanismRegistry with lifecycle tracking
- ~~**K4** (full settlement)~~ RESOLVED — KernelDAGRunner with topological phase ordering
- ~~**K5** (metrics + judge)~~ RESOLVED — round_metrics + judge_results + JudgeAdapter

### ~~Should fix before Phase 4 (multi-tenant / federation)~~ ALL RESOLVED
- ~~**K7** (receipt signing)~~ RESOLVED — Ed25519 oracle identity, receipts signed
- ~~**K8** (federation fields)~~ RESOLVED — Optional fields in RoundReceipt
- ~~**K9** (constitution)~~ RESOLVED — 3-layer Constitution with hash in receipts
- ~~**A4** (event envelope enrichment)~~ RESOLVED — world_id, phase, payload_hash, prev_event_hash

### ~~Should fix before production — persistence~~ ALL RESOLVED
- ~~**A2** (SQLite event log)~~ RESOLVED — `kernel/store.py` + `kernel/event_log.py`, WAL mode
- ~~**A5** (CORS)~~ RESOLVED — CORSMiddleware added
- ~~**A7** (memory eviction)~~ RESOLVED — Events backed by SQLite
- ~~**P3-4** (mechanism persistence)~~ RESOLVED — MechanismRegistry uses Store
- ~~**P4-2** (snapshot persistence)~~ RESOLVED — ModeratorState uses Store

### Fix when convenient (low urgency)
- ~~**K2** (richer snapshots)~~ RESOLVED — age, productivity, strength, consumption, overall_productivity
- **K10** (save_path coupling) — cosmetic, works fine
- **K12** (messages_sent) — wire when messaging matters
- **A1** (multi-world) — add alongside, no breaking change
- ~~**A3** (SSE)~~ RESOLVED — `GET /v1/world/events/stream` with sse-starlette
- **A8** (OpenAPI pinning) — add before SDK release
- ~~**A9** (headless sim)~~ RESOLVED — Phase 2 write path adds submission window

---

## Phase 3: Governance

### P3-1. JudgeAdapter timeout is hardcoded

- **What we built:** 30s timeout, not configurable at runtime.
- **Risk:** Different proposal types may need different timeouts.
- **When to fix:** When dynamic judge configuration is needed.

### P3-2. No LLM cost tracking per judgment

- **What we built:** JudgeAdapter doesn't track token usage or cost per judgment.
- **Risk:** No visibility into judge operational costs.
- **When to fix:** When cost budgeting for judge operations matters.

### P3-3. Gini coefficient only over vitality

- **What we built:** Gini computed over member vitality only. Doesn't consider cargo, land, or total wealth.
- **Risk:** Inequality metric is one-dimensional.
- **When to fix:** When a more nuanced inequality metric is needed.

### ~~P3-4. MechanismRegistry is in-memory only~~ RESOLVED

- **Resolved:** 2026-02-24. `MechanismRegistry` now accepts a `Store` (SQLite). All mutations (submit, approve, reject, activate) are persisted via `upsert_mechanism()`. Registry loads existing records on init. Shares the same `leviathan.db` with event log (A2).

### P3-5. No mechanism rollback or deactivation

- **What we built:** Once active, mechanisms are permanent. No way to deactivate or roll back.
- **Risk:** Bad mechanisms can't be undone.
- **When to fix:** Phase 4 governance features.

### P3-6. Single-judge LLM evaluation

- **What we built:** One LLM call per proposal. No multi-judge consensus, appeal, or human override.
- **Risk:** Single point of failure for governance decisions.
- **When to fix:** When trust guarantees require multi-party validation.

### P3-7. No mechanism versioning

- **What we built:** Can't update an active mechanism. Must propose a new one.
- **Risk:** No evolution path for existing mechanisms.
- **When to fix:** When mechanism evolution patterns emerge.

---

## Phase 4: Constitution, Signing & Moderator

### P4-1. Oracle private key in-memory only

- **What we built:** Ed25519 private key held in Python memory. Lost on restart, no key rotation.
- **Risk:** World oracle identity changes after restart. No key persistence.
- **When to fix:** Before production. Add encrypted key file or vault integration.

### ~~P4-2. Rollback limited to in-memory snapshot history~~ RESOLVED

- **Resolved:** 2026-02-24. `ModeratorState` now persists snapshots to SQLite `snapshots` table via `Store`. Eviction keeps last N snapshots (default 10). Snapshots survive restarts.

### P4-3. No moderator action rate limiting

- **What we built:** Moderator endpoints have no separate rate limiting.
- **Risk:** Compromised moderator key could spam pause/resume/ban.
- **When to fix:** When multi-moderator support is needed.

### P4-4. Constitution amendments not versioned as a chain

- **What we built:** Version counter increments but no history of previous states.
- **Risk:** No diff between constitution versions, no audit trail of amendments.
- **When to fix:** When governance audit trail matters.

### P4-5. Ban is binary (no graduated penalties)

- **What we built:** Agents are either fully banned or fully active.
- **Risk:** Cannot partially restrict misbehaving agents (e.g., read-only mode).
- **When to fix:** When more nuanced moderation is needed.

### P4-6. Federation fields reserved but not populated

- **What we built:** Optional fields in RoundReceipt and EventEnvelope, all None.
- **Risk:** No actual cross-world communication or verification.
- **When to fix:** Phase 5 (federation implementation).
