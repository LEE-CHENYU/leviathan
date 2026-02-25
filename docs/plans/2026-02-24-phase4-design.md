# Phase 4: Constitution, Receipt Signing, Federation Prep & Moderator APIs — Design

Date: 2026-02-24
Status: Approved
Builds on: Phase 0 (kernel) + Phase 1 (read API) + Phase 2 (write API) + Phase 3 (governance)

## Goal

Establish a formal trust model (Constitution), cryptographic identity (Ed25519 oracle signing), federation-ready receipt/event schemas, and moderator override APIs — completing the single-world trust and administrative layer before federation.

## Decisions Made

- **Implementation order**: Constitution-first (defines what moderators can/can't do, and what receipts sign over)
- **Moderator auth**: Superuser API key with elevated privileges (extends existing APIKeyAuth)
- **Constitution storage**: Python dataclass + YAML file, SHA-256 hash in receipts
- **Receipt signing**: Ed25519 keypair per world (deterministic from seed when provided)
- **Federation fields**: Reserved as Optional in RoundReceipt + EventEnvelope (populated in Phase 5)

## Component 1: Constitution Object (resolves K9)

### Problem

No formal specification of what invariants the world guarantees. Makes trust/audit harder. Design docs prescribe a 3-layer constitutional model.

### Solution

A `Constitution` dataclass with three clause layers of decreasing rigidity:

```python
@dataclass
class Constitution:
    kernel_clauses: Dict[str, str]      # Immutable, hardcoded truths
    governance_clauses: Dict[str, str]   # Amendable via judge-approved mechanisms
    world_rules: Dict[str, str]          # Open, changeable by any approved mechanism
    version: int                         # Increments on any governance/world_rules change
```

### Kernel Clauses (immutable)

- `"energy_conservation"`: "Total vitality cannot be created from nothing"
- `"identity_permanence"`: "Agent IDs are permanent and unique"
- `"determinism"`: "Same seed + same actions = same outcome"

### Governance Clauses (amendable via judge approval)

- `"max_tax_rate"`: "No mechanism may extract more than 50% of an agent's vitality"
- `"proposal_limit"`: "One mechanism proposal per agent per round"

### World Rules (open)

- `"trade_fee"`: "5% vitality fee on all trades"
- `"land_expansion_cost"`: "Expanding costs 10 vitality"

### Storage & Loading

- Default constitution defined in `kernel/constitution_default.yaml`
- Loaded at `WorldKernel.__init__()` — if no custom YAML path, use defaults
- `constitution_hash` = SHA-256 of canonical JSON (sorted keys, compact separators)
- Hash included in every `RoundReceipt`

### Amendment Flow

1. Agent proposes a mechanism that modifies governance or world rules
2. JudgeAdapter evaluates the proposal including constitution context
3. If approved, the change updates `Constitution.governance_clauses` or `world_rules`
4. `Constitution.version` increments
5. New `constitution_hash` reflected in next receipt
6. Kernel clauses cannot be changed — any mechanism attempting to modify them is rejected

### Files

- Create: `kernel/constitution.py` — Constitution dataclass, load/save/hash
- Create: `kernel/constitution_default.yaml` — default clauses
- Modify: `kernel/schemas.py` — add `constitution_hash` to RoundReceipt
- Modify: `kernel/world_kernel.py` — load constitution, include hash in receipts

---

## Component 2: Receipt Signing & Federation Prep (resolves K7, K8, A4)

### Oracle Identity (K7)

Each world gets an Ed25519 keypair at creation:

```python
@dataclass
class OracleIdentity:
    world_public_key: str    # hex-encoded Ed25519 public key
    _private_key: bytes      # kept in memory, never serialized

    def sign(self, data: bytes) -> str:
        """Sign data, return hex-encoded signature."""

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify a hex-encoded signature."""

    @classmethod
    def generate(cls) -> "OracleIdentity":
        """Generate a new Ed25519 keypair."""

    @classmethod
    def from_seed(cls, seed: int) -> "OracleIdentity":
        """Deterministic keypair from seed (for reproducible worlds)."""
```

- Generated in `WorldKernel.__init__()` — deterministic from `random_seed` if provided
- `world_public_key` exposed via `GET /v1/world` snapshot
- Private key never leaves the kernel process

### Receipt Signing

New fields on `RoundReceipt`:

```python
oracle_signature: Optional[str] = None     # hex Ed25519 signature
constitution_hash: Optional[str] = None    # SHA-256 of constitution state
world_public_key: Optional[str] = None     # identifies the signing oracle
```

Signing flow in `settle_round()`:
1. Build receipt with `oracle_signature=None`
2. Serialize to canonical JSON (sorted keys, compact separators)
3. Sign canonical bytes with oracle private key
4. Set `oracle_signature` on the receipt

### Federation-Prep Fields in Receipt (K8)

Reserved as `Optional[...]` fields on `RoundReceipt`, all `None` until federation:

```python
origin_world_id: Optional[str] = None
origin_receipt_hash: Optional[str] = None
bridge_channel_id: Optional[str] = None
bridge_seq: Optional[int] = None
notary_signature: Optional[str] = None
```

### Event Envelope Enrichment (A4)

New `Optional` fields on `EventEnvelope`:

```python
world_id: Optional[str] = None
phase: Optional[str] = None
payload_hash: Optional[str] = None
prev_event_hash: Optional[str] = None
```

### Files

- Create: `kernel/oracle.py` — OracleIdentity, sign/verify, keypair generation
- Modify: `kernel/schemas.py` — add signature + federation + constitution fields to RoundReceipt
- Modify: `kernel/world_kernel.py` — create oracle, sign receipts, expose public key
- Modify: `api/models.py` — enrich EventEnvelope model
- Modify: `api/routes/world.py` — include `world_public_key` in snapshot response

### Dependency

Uses `cryptography` library (`cryptography.hazmat.primitives.asymmetric.ed25519`).

---

## Component 3: Moderator APIs

### Moderator Auth Model

Extend `APIKeyAuth` to support role-based keys:

```python
class APIKeyAuth:
    def __init__(self, api_keys=None, moderator_keys=None):
        self.api_keys = api_keys or set()
        self.moderator_keys = moderator_keys or set()
        self.enabled = bool(self.api_keys or self.moderator_keys)
```

New `require_moderator` dependency:
```python
def require_moderator(request: Request) -> None:
    """Raises 403 if the caller is not using a moderator API key."""
```

Regular API keys access Phase 1-3 endpoints. Only moderator keys unlock admin endpoints.

### Moderator Endpoints

| Endpoint | Method | Effect |
|----------|--------|--------|
| `/v1/admin/pause` | `POST` | Pause simulation (no new rounds start) |
| `/v1/admin/resume` | `POST` | Resume simulation |
| `/v1/admin/ban/{agent_id}` | `POST` | Ban agent (actions/proposals rejected) |
| `/v1/admin/unban/{agent_id}` | `POST` | Unban agent |
| `/v1/admin/rollback` | `POST` | Roll back to specific round |
| `/v1/admin/quotas` | `PUT` | Adjust per-agent rate limits or caps |
| `/v1/admin/status` | `GET` | Moderator dashboard |

### Pause/Resume

- `RoundState` gets a `paused: bool` flag
- `_simulation_loop` checks `round_state.paused` at top of each iteration
- Pausing doesn't interrupt a round in progress — prevents next round from starting
- Event logged: `{"event_type": "admin_pause", "moderator": "key_hash", ...}`

### Ban/Unban

- `AgentRegistry` gets a `banned_agents: Set[int]` set
- Banned agents' action submissions return 403
- Banned agents' mechanism proposals rejected before reaching judge
- Event logged: `{"event_type": "admin_ban", "agent_id": ..., "moderator": ...}`

### Rollback

- Stores last N round snapshots in memory (configurable, default 10)
- Rollback resets `WorldKernel._round_id`, restores snapshot state, clears caches
- Event logged with before/after state hashes
- Cannot violate kernel clauses (rollback itself is deterministic)

### Quotas

- Per-agent action rate limits (max actions per round, default unlimited)
- Per-agent proposal limits (already 1/round, configurable)
- Stored in a `ModeratorConfig` dataclass in app state

### Audit Trail

Every moderator action emits an `EventEnvelope` with:
- `event_type`: `"admin_*"` prefix
- `payload`: includes `moderator_key_hash` (SHA-256 of key, not the key itself), action details

### Files

- Create: `api/routes/admin.py` — all moderator endpoints
- Create: `kernel/moderator.py` — ModeratorConfig, snapshot history, rollback logic
- Modify: `api/auth.py` — moderator_keys, require_moderator
- Modify: `api/round_state.py` — paused flag
- Modify: `api/registry.py` — banned_agents, is_banned()
- Modify: `api/routes/actions.py` — ban check
- Modify: `api/routes/mechanisms.py` — ban check
- Modify: `api/app.py` — include admin router, moderator_keys
- Modify: `api/deps.py` — moderator config to app state
- Modify: `scripts/run_server.py` — moderator-keys CLI arg, pause check

---

## Component 4: Tech Debt Documentation

### Mark Resolved

- **K7** (receipt signing) → RESOLVED: Ed25519 oracle identity, receipts signed
- **K8** (federation fields) → RESOLVED: Optional fields in RoundReceipt
- **K9** (constitution) → RESOLVED: 3-layer Constitution with hash in receipts
- **A4** (event envelope) → RESOLVED: world_id, phase, payload_hash, prev_event_hash added

### New Phase 4 Compromises

| ID | Compromise | Risk | When to Fix |
|----|-----------|------|-------------|
| P4-1 | Oracle private key in-memory only (no HSM/vault) | Key lost on restart, no key rotation | Before production |
| P4-2 | Rollback limited to in-memory snapshot history (last N rounds) | Can't rollback beyond N rounds | When SQLite lands (A2) |
| P4-3 | No moderator action rate limiting | Compromised key could spam admin endpoints | Multi-moderator phase |
| P4-4 | Constitution amendments not versioned as a chain | No diff between versions | When governance audit trail matters |
| P4-5 | Ban is binary (no graduated penalties) | Can't partially restrict agents | When nuanced moderation needed |
| P4-6 | Federation fields reserved but not populated | No cross-world communication | Phase 5 (federation) |

---

## Testing Strategy

### Constitution tests (~8 tests)
- Load from YAML, default clauses present
- constitution_hash is deterministic (same clauses = same hash)
- Amending governance clause increments version + changes hash
- Amending kernel clause raises error
- World rules amendment works
- Hash included in receipt
- Load custom YAML overrides defaults
- Empty constitution has valid hash

### Oracle/Signing tests (~6 tests)
- Generate keypair, public key is hex string
- Deterministic keypair from same seed
- Sign data, verify with public key succeeds
- Tampered data fails verification
- Receipt signature is verifiable
- Different seeds produce different keys

### Moderator API tests (~12 tests)
- Pause/resume toggles state
- Sim loop respects paused flag
- Ban blocks agent action submissions (403)
- Ban blocks mechanism proposals (403)
- Unban re-enables agent
- Rollback restores prior snapshot
- Rollback resets round_id
- Quotas enforce per-agent limits
- Admin status returns correct state
- Non-moderator key gets 403 on admin endpoints
- Every admin action emits an event
- Moderator key_hash in event payload (not raw key)

### Federation field tests (~4 tests)
- New receipt fields present (all None initially)
- Event envelope has enriched fields
- Event payload_hash is correct
- prev_event_hash chains events

### Integration tests (~2 tests)
- Full moderator workflow: pause → ban → rollback → resume → verify events
- Constitution amendment: mechanism proposal → judge approval → updated hash in receipt

---

## Files Changed Summary

### New files (5)
- `kernel/constitution.py`
- `kernel/constitution_default.yaml`
- `kernel/oracle.py`
- `kernel/moderator.py`
- `api/routes/admin.py`

### Modified files (12)
- `kernel/schemas.py` — RoundReceipt gets signature + federation + constitution_hash fields
- `kernel/world_kernel.py` — oracle identity, constitution, sign receipts
- `api/auth.py` — moderator_keys, require_moderator
- `api/models.py` — EventEnvelope enrichment, admin request/response models
- `api/deps.py` — moderator config to app state
- `api/app.py` — include admin router, moderator_keys
- `api/round_state.py` — paused flag
- `api/registry.py` — banned_agents, is_banned()
- `api/routes/actions.py` — ban check
- `api/routes/mechanisms.py` — ban check
- `scripts/run_server.py` — moderator-keys CLI arg, pause check
- `docs/plans/2026-02-24-implementation-compromises.md` — mark K7-K9, A4 resolved + Phase 4 compromises

### Existing files unchanged
- `kernel/mechanism_registry.py` — untouched
- `kernel/judge_adapter.py` — untouched
- `kernel/round_metrics.py` — untouched
- `kernel/dag_runner.py` — untouched
- `kernel/subprocess_sandbox.py` — untouched
