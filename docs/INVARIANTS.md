# Leviathan System Invariants

**Status:** Authoritative, version-independent
**Scope:** Every implementation, deployment, refactor, and extension must respect these rules
**Authority:** Changes to Layer 0 require explicit operator approval and migration; they cannot be introduced silently

This document is the single source of truth for what must never break. It consolidates invariants from the constitutional model (`docs/system-design/05-meta-edit.md`), kernel constitution (`kernel/constitution_default.yaml`), and engineering constraints (`CLAUDE.md`).

---

## Layer 0: Kernel Invariants (Unamendable)

These are the trusted computing base. If any of these break, the world loses meaning as a shared, auditable object. No in-world agent, mechanism, or governance change may violate them.

### I-1. Deterministic Replay

> Same seed + same accepted events = same state hash.

- Settlement phases must use only seeded PRNG, no wall-clock time, no network calls, no unseeded randomness.
- The DAG phase order (`new_round -> analyze -> ... -> log_status`) is a protocol specification, not an implementation detail. It must not be reordered.
- `state_hash` in `RoundReceipt` must be reproducible by any correct implementation given the same inputs.

### I-2. Event Log Integrity

> The event log is append-only. No event may be altered or deleted after creation.

- Events are ordered, sequenced, and hashed.
- Each `RoundReceipt` includes `snapshot_hash_before`, `snapshot_hash_after`, and `constitution_hash`.
- Receipts are signed by the oracle (`oracle_signature`). Signature verification must be possible by any observer holding `world_public_key`.
- Deletion or mutation of committed events is forbidden, even by moderators.

### I-3. Safety Boundary

> No untrusted code executes in the oracle process.

- Agent-submitted code and mechanism code run in a sandboxed subprocess with timeout and resource limits.
- The sandbox communicates with the kernel only via a serialized action proxy (JSON intents).
- The oracle applies state transitions; agents submit intents. These roles never merge.

### I-4. Role Separation

> Players cannot directly mutate world state.

- Only the oracle applies state transitions (via `WorldKernel`).
- The judge execution environment is independent of players.
- Moderator controls (pause, rollback, ban) cannot be disabled by gameplay.

### I-5. Energy Conservation

> Total vitality cannot be created from nothing.

- Vitality enters the system through `produce()` (resource extraction from land).
- Vitality leaves the system through `consume()` (survival cost).
- No mechanism may create vitality without a conserved source.

### I-6. Identity Permanence

> Agent IDs are permanent and unique within a world instance.

- Once assigned, an agent ID is never reused, even if the member dies.
- The mapping `agent_id -> member_id` is immutable for the lifetime of a registration.

---

## Layer 1: Engineering Invariants (Must Hold Across All Deployments)

These are not game rules but engineering contracts that ensure the system works correctly under all operational conditions.

### E-1. State Persistence

> All authoritative state must survive process restarts.

Authoritative state includes:
- **Event log** — the complete history of round settlements (receipts, metrics, judge results)
- **Mechanism registry** — the full lifecycle of every proposal (submitted, approved, rejected, active)
- **Oracle identity** — the Ed25519 keypair that signs receipts (must be deterministic from seed, or persisted)
- **Rollback snapshots** — recent world snapshots for moderator rollback

Non-authoritative (ephemeral) state that may be lost:
- Agent registration sessions (agents can re-register)
- Rate limiter counters
- In-flight submission window state

**Implementation rule:** Use a durable store (SQLite) for authoritative state. In-memory caches may exist but must be backed by the durable store. The `LEVIATHAN_DATA_DIR` must point to persistent storage in production.

### E-2. Atomic Writes

> State mutations must be atomic — either fully applied or not applied.

- Database writes use transactions.
- File writes use write-to-temp + atomic rename.
- A crash mid-write must not corrupt the store.

### E-3. Single Writer

> The simulation loop is the sole writer of round state.

- One simulation thread advances the world. API handlers only read state or enqueue submissions.
- No concurrent writers to the same database table for round-advancing operations.
- Read-only API queries may run concurrently with writes (WAL mode).

### E-4. Bounded Memory

> In-memory data structures must not grow unboundedly.

- Event history is backed by persistent storage; in-memory cache has a maximum size.
- Rollback snapshots have a maximum count.
- Mechanism records are persisted; in-memory is a cache.

### E-5. Deployment Continuity

> A deploy, restart, or scale event must not silently discard world history.

- Persistent volumes must be mounted before the application starts.
- The application must verify the data directory is writable on startup.
- If the data directory is missing or empty, the application starts a fresh world (this is fine). But it must never start a fresh world while an existing database is present on a different path (split-brain).

---

## Layer 2: Governance Invariants (Amendable with Guardrails)

These can be changed through in-world governance, but changes require judge approval and are subject to Layer 0 constraints.

### G-1. Mechanism Lifecycle

- One mechanism proposal per agent per round.
- Proposals must pass judge evaluation before activation.
- Activated mechanisms are recorded in the event log.

### G-2. Tax Limits

- No mechanism may extract more than 50% of an agent's vitality in a single round.

### G-3. Judge Independence

- Judge policy can be amended by governance.
- Judge infrastructure (the evaluation service itself) cannot be replaced by in-world action.

---

## Enforcement

- **Tests:** Each invariant should have at least one test that asserts it. Test names should reference the invariant ID (e.g., `test_I1_deterministic_replay`).
- **Code review:** Any PR that touches kernel, oracle, or persistence code must be checked against this document.
- **Constitution hash:** The `constitution_hash` in every `RoundReceipt` cryptographically anchors the governance state to the event log.
- **This document:** Lives at `docs/INVARIANTS.md`. Changes to Layer 0 require a migration notice in the changelog. Changes to Layer 1 require updating the implementation compromises doc.
