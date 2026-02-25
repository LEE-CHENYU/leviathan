I read the three notes as one coherent “design packet”:
	•	(1) Leviathan Current State Audit — what you have today and what’s missing.
	•	(2) Target Distributed Architecture — what “open, agent-discoverable world” should look like.
	•	(3) Implementation Blueprint — how to get there without breaking semantics.

Below is a deep brainstorm that tries to (a) elaborate what’s already implied in each section, (b) surface risks/tradeoffs that usually bite later, and (c) propose concrete design moves that preserve what’s special about your repo (deterministic rounds + mechanism co-design + rich eval history).

⸻

1) The core story I see

You already have a pretty rare combo:
	•	A deterministic-ish round-based simulation with explicit phases wired through a DAG graph engine.
	•	Agents that don’t just “act,” but can propose mechanisms and code (and you have scaffolding for judge + contracts + physics constraints).
	•	Real observability via execution histories and metrics aggregation.

The big missing piece is that everything is still single-process, in-process, and trust-blind. That’s fine for research and internal experiments, but the minute you allow external agents to participate, your weakest links become:
	1.	Execution safety (in-process exec is a hard no once you’re on the internet).
	2.	Role separation (players can’t be too close to oracle/judge).
	3.	Protocol surface + discovery (agents need a machine-readable “how do I join / what can I do / what are the deadlines / what’s the schema?”).
	4.	Event sourcing / receipts (auditability + replay becomes the backbone once you distribute).

Your “Target Distributed Architecture” and “Blueprint” correctly aim at those four.

⸻

2) Deep thoughts on the Current State Audit

2.1 Two-layer simulation structure is already a strong modular boundary

You have:
	•	Leviathan/ as base sim kernel
	•	MetaIsland/ as LLM-driven extension layer (code proposals, judge, contracts, physics, graph execution)

Even if today they’re coupled in-process, conceptually this separation is exactly what you want long-term:
	•	WorldKernel (oracle) = pure-ish, deterministic state transition.
	•	Agent layer = untrusted, external, bring-your-own-compute.
	•	Governance layer = judge/moderation and policy.

The main refactor is to turn the conceptual boundary into a process boundary and a data boundary.

2.2 Your phase DAG is more than orchestration — it’s a protocol spec hiding in code

You listed the default wiring:

new_round -> analyze -> propose_mechanisms -> judge -> execute_mechanisms -> agent_decisions -> execute_actions -> contracts -> produce -> consume -> environment -> log_status

In distributed terms, this becomes:
	•	A round lifecycle with windows.
	•	Clear inputs and outputs per phase.
	•	Clear “who is allowed to write” in each phase.

Right now the phases are internal function calls. Your docs correctly propose turning them into externally visible phase deadlines. That’s not just operational—it’s a fairness primitive.

2.3 Your metrics tell a story: governance “accepts,” but execution “doesn’t land”

From the audit summary:
	•	attempted mechanisms: 3
	•	approved: 3
	•	executed: 1
	•	high action-signature diversity
	•	negative survival delta

A few hypotheses worth testing (because they imply different architectural priorities):
	1.	Mechanism proposals are getting approved but failing at integration
That suggests the governance pipeline isn’t yet strongly coupled to settlement semantics (exactly what you note: physics constraints not fully enforced; environment node reports counts but doesn’t enforce deterministically).
	2.	Mechanisms execute, but their effects aren’t observable / persisted correctly
Then you need “receipts” not just for audit, but to debug why only 1 executed.
	3.	The world is too harsh / unstable
The negative survival delta could be emergent behavior, but it could also indicate missing invariants: conservation constraints, anti-suicide incentives, or runaway resource sinks. In an open world, this becomes a moderation/governance problem (agents will exploit collapse dynamics).

If you want “open participation,” stabilizing the sim economy and survival dynamics isn’t just gameplay polish—it prevents adversarial griefing. A world that collapses easily is the easiest target.

2.4 The gaps you list are the real blockers; here’s what they imply concretely

(A) “No external protocol or discovery surface”
This means today there’s no machine contract for:
	•	how to join
	•	what actions exist
	•	what fields are allowed
	•	what timing rules exist

Without this, you can’t have “one-shot onboarding.” You’ll end up with bespoke client code and humans in the loop. Your .well-known + OpenAPI direction is the right fix.

(B) “Code execution trust boundary too weak”
This is existential. In open participation:
	•	agent “action code”
	•	mechanism code
	•	contract code

…cannot run in the oracle process.

Even if you sandbox, you should assume sandbox escape attempts. The oracle must be able to survive malicious submissions by:
	•	rejecting early via schema validation and static checks,
	•	isolating dynamic checks in a worker,
	•	applying only “safe, validated deltas” to state.

(C) “Governance roles not separated enough”
This isn’t just architecture purity. It’s incentive alignment and legitimacy:
	•	If players believe the judge can be influenced or that the oracle can “cheat,” participation drops.
	•	Even if you’re not fully decentralized, you need credible neutrality through separation + audit logs.

(D) “Mechanism lifecycle incomplete”
You’re right to call out versioning, staged rollout, rollback. In open worlds, rule changes are deployments, not “code snippets.”

A mechanism proposal lifecycle should look like:
	•	submitted → validated → simulated → judged → canary → active → rollback-able

And each transition should emit an event, so observers can audit.

(E) “Reproducibility drift”
Your mention of binary mismatch already points to a key operational reality:
	•	If you want third-party agents and contributors, you need deterministic builds and pinned environments.
	•	Otherwise you’ll get “works on my machine” divergence that destroys replay trust.

⸻

3) Thoughts on the Target Distributed Architecture

3.1 Product goals are coherent, but one implicit goal matters: agent ergonomics

Your discovery layer is a great start. I’d explicitly think of it as an LLM-friendly protocol:
	•	Provide both:
	•	machine-readable schemas (OpenAPI + JSON Schema),
	•	and LLM-usable onboarding text (“onboarding prompt” endpoint).

This is actually a big differentiator: most APIs are machine-readable but not agent-readable. You want both.

3.2 Role model: the “hard rule” is correct, but you’ll want cryptographic reinforcement

You state:

Player Agent cannot mutate world state directly. Only the oracle applies state transitions.

In practice, I’d reinforce this with:
	•	All state transitions are derived from an append-only log of accepted intents.
	•	Each accepted intent is:
	•	signed by the agent,
	•	timestamped (or round-window bounded),
	•	hashed into the round receipt.

This isn’t “blockchain”—it’s basic tamper-evidence.

3.3 Discovery endpoints: good, but they should include version negotiation + minimal “hello world”

Your list:
	•	/.well-known/leviathan-agent.json
	•	/openapi.json
	•	/v1/protocol/capabilities
	•	/v1/protocol/onboarding-prompt

I’d add (or embed in capabilities):
	•	protocol_version and min_supported_version
	•	a canonical example action and example response (small but huge for agent onboarding)
	•	a pointer to JSON Schemas for:
	•	ActionIntent
	•	MechanismProposal
	•	RoundReceipt
	•	EventEnvelope

Agents do better when they can copy a template.

3.4 Control plane: this is where sybil resistance and abuse prevention live

You list identity registration, key mgmt, quotas, tenancy, capability negotiation. Two additional design thoughts:
	1.	Don’t overfit to “human user accounts.”
External agents may be ephemeral. Your flow suggests ephemeral keypair → request token, which is good.
	2.	Quotas and admission control aren’t optional in open worlds.
Even with “good actors,” you’ll see accidental overload. With bad actors, you’ll see:
	•	join spam
	•	huge payload spam
	•	deadline DDoS (submit storms right before cutoff)
	•	replay attacks

So you’ll want:
	•	strict payload size caps,
	•	per-world and per-agent rate limits,
	•	idempotency keys (you mention),
	•	and probably some friction for registration in open beta (even if lightweight).

3.5 Oracle plane: determinism is a property you have to design for, not “test later”

You’re already thinking this way (great). Key pitfalls that commonly break determinism once distributed:
	•	Non-canonical serialization
If you hash JSON, you must canonicalize key ordering and numeric formats.
	•	Floating-point and platform differences
If any mechanism sandbox uses floats, replay across machines can diverge.
	•	Iteration order issues
Python dict ordering is stable within a process but becomes subtle across versions and when you build composite data structures.
	•	Time and randomness
Oracle must not read system time, network, or any nondeterministic source during settlement.

Your blueprint’s “receipt object with seed + state hash” is the right direction, but I’d push further:
	•	Define a single canonical hashing and encoding method early.
	•	Treat it as part of the protocol spec.

3.6 Event and storage plane: event sourcing is the right primitive, but define event semantics early

You say:
	•	append-only event log source of truth,
	•	materialized state for reads,
	•	immutable round receipts.

Yes.

What I would elaborate is: the event taxonomy becomes your real API.
Endpoints come and go; events persist.

So define:
	•	event envelope with fields like:
	•	world_id, seq, timestamp, round_id, phase, type, payload, hash_prev, hash_self (or at least receipt-level hashing)
	•	strict phase ordering rules

This becomes the “ledger” without needing chains.

3.7 Round lifecycle: your phases are excellent; commit-reveal is powerful but should be modular

Your lifecycle:
	1.	snapshot_open
	2.	analysis_window
	3.	mechanism_proposal_window
	4.	judge_window
	5.	action_commit_window (optional)
	6.	action_reveal_window (optional)
	7.	settlement
	8.	receipt_publish

This is conceptually solid. The big tradeoff:
	•	Commit-reveal increases fairness (reduces last-mover advantage and reactive exploitation).
	•	Commit-reveal increases complexity and failure modes (agents fail to reveal, network hiccups, etc.).

Your doc already says “optional.” I’d formalize that as a capability:
	•	World declares: supports_commit_reveal: true/false
	•	Agent declares: supports_commit_reveal: true/false

Then the oracle can run either:
	•	simple lockstep, or
	•	commit-reveal.

Also: if you do commit-reveal, specify the failure policy:
	•	If agent commits but doesn’t reveal → forfeit actions that round (and maybe penalty).
	•	If reveal comes late → reject.

This is gameplay + protocol.

3.8 Mechanism runtime: DSL-first is usually the right v1, but not for the reason people think

You propose:
	•	typed DSL / constrained AST subset preferred,
	•	optional sandbox runtime (WASM or jailed Python worker)

My take:
	•	DSL-first is less about security (though it helps),
	•	and more about governance and evaluability.

If your mechanism changes are expressed as:
	•	parameter changes,
	•	rule additions with explicit invariants,
	•	limited types of state mutation,

…then your judge can reason better, your canary tests are faster, and you can build a stable ecosystem sooner.

Sandboxed Python is attractive for expressiveness, but it tends to explode:
	•	complexity,
	•	nondeterminism,
	•	covert channels (resource timing),
	•	and review difficulty.

A pragmatic hybrid:
	•	Start with “mechanism templates” (predefined mechanism classes with parameters).
	•	Then allow constrained AST subset for more flexibility.
	•	Then add sandboxed code as an “advanced” capability later.

⸻

4) Thoughts on the Implementation Blueprint

4.1 Phase 0: “WorldKernel extraction” is the keystone

If Phase 0 is done right, everything else is easier.

What “done right” means in practice:
	•	The oracle exposes a function like:
(state_before, round_inputs, active_mechanisms, seed) -> (state_after, events_emitted, receipt)
	•	Each phase is a typed input, not “call some Python.”
	•	The kernel has no network calls, no LLM calls, no filesystem side effects.

If you achieve that, then:
	•	FastAPI wrapping becomes boring plumbing,
	•	event replay becomes natural,
	•	sandboxing becomes additive rather than invasive.

4.2 “Freeze and document current action schema” — do it as JSON Schema + generated types

This is one of those “small sounding” tasks that determines your velocity.

If you define:
	•	ActionIntent JSON Schema
	•	Response and error schemas

Then you can:
	•	generate SDK types,
	•	validate payloads at the edge,
	•	fuzz test inputs,
	•	and keep compatibility.

4.3 Phase 1: Read API + event feed is a smart first external surface

Read-only endpoints let you test:
	•	discovery,
	•	auth-free observation,
	•	snapshot serialization,
	•	event ordering.

One thought: don’t underestimate event feed ergonomics.
Agents and observers often want:
	•	SSE (server-sent events),
	•	or websocket,
	•	or long-poll with after=seq.

Your doc uses GET /events?after=... — good baseline. I’d ensure you have:
	•	stable monotonic per-world sequence,
	•	consistent pagination,
	•	bounded payload sizes,
	•	and optional filtering by round/phase/type.

4.4 Phase 2: External write path — idempotency keys + deadline enforcement are the “distributed correctness” core

This phase introduces distributed failure realities:
	•	Clients retry.
	•	Requests arrive out of order.
	•	Deadlines are raced.

The blueprint already includes:
	•	reject late submissions
	•	idempotency keys

Two extra things I’d add:
	1.	Server-generated action IDs + deterministic acceptance rules
You want acceptance to be deterministic given the event log. That means:
	•	actions are stored with a stable ID,
	•	acceptance emits an event with reason codes for rejections.
	2.	Clock-skew strategy
Deadlines based on server time. Clients can be skewed. Provide:
	•	server time in round status,
	•	maybe allow a small grace window but record lateness.

4.5 Phase 3: Judge service — treat it as an “audited compiler + policy engine”

Your judge service responsibilities:
	•	static checks,
	•	deterministic lint,
	•	LLM reasoning checks.

I’d expand the framing:
	•	Static validation: schema + forbidden constructs + bounded complexity.
	•	Semantic simulation: run mechanism in a sandbox against test scenarios + invariant suite.
	•	Policy scoring: fairness/exploit resistance signals, ideally structured.
	•	Decision: approve/reject + scope.

Crucially: judge outputs must be stored as receipts so replay doesn’t require rerunning the LLM.

This also gives you governance legitimacy: observers can inspect judge receipts.

4.6 Phase 4: Moderator & multi-tenant — build operationally minimal but correct

Moderator APIs:
	•	pause world
	•	force rollback
	•	ban session
	•	adjust quotas

These are essential for open beta. Even a benign bug in mechanism execution becomes catastrophic without a pause + rollback.

Also: “tenant isolation” is not just quotas. You need to make sure:
	•	one world’s agent spam doesn’t degrade all worlds,
	•	one mechanism proposal queue doesn’t block judge throughput globally.

4.7 Security baseline: strong list; the missing piece is “canonical hashing strategy”

Your baseline is good:
	1.	signed requests + replay protection
	2.	payload validation
	3.	no exec in oracle
	4.	sandbox limits
	5.	append-only audit log

One more must-have, because you keep emphasizing determinism and audit:
	6.	Canonical serialization + hashing spec for receipts/snapshots/events.

If you get that wrong later, you’ll have painful migrations.

4.8 Testing plan: you’re already thinking like a distributed systems team

I like your phase-gated tests. I’d add two “gotcha” test categories:
	•	Adversarial protocol tests
	•	duplicate submissions with same idempotency key
	•	late reveals (commit-reveal)
	•	payload bombs (near max size)
	•	invalid signatures, signature replay
	•	Metamorphic / property-based sim tests
	•	invariants like conservation (if expected)
	•	monotonic constraints (if any)
	•	“no negative resources”
	•	“population can’t increase without reproduction action” (example)

These tests catch emergent exploit vectors.

⸻

5) The biggest design decisions hiding in plain sight

Your blueprint lists decision points; here’s how I’d think about them.

5.1 Mechanism language strategy: DSL-first vs sandbox-first

DSL-first advantages
	•	easier static analysis
	•	easier deterministic replay
	•	easier judge reasoning
	•	smaller attack surface

Sandbox-first advantages
	•	faster expressiveness, fewer constraints on creativity

My bias for open participation:
	•	Start with mechanism templates + param changes + restricted AST (like “safe patch types”).
	•	Add sandboxed code as an opt-in advanced capability once you have:
	•	stable receipts,
	•	robust canary/rollback,
	•	resource isolation.

5.2 Identity: API keys vs signed keypairs

If you want “bring-your-own agent compute” and credible audit trails, signed requests become very natural:
	•	Agent generates keypair.
	•	Register public key.
	•	Server issues a session token (short-lived).
	•	Requests are signed + include nonce / timestamp / round_id.

This is also cleaner than long-lived API keys floating around.

5.3 Fairness: lockstep deadlines vs commit-reveal

For a research sandbox, lockstep is fine.
For competitive or adversarial participation, commit-reveal is a major fairness upgrade.

A good compromise:
	•	Make commit-reveal a world-level config capability.
	•	Start with lockstep for v0, but ensure your schemas already have the fields to support commit later (so you don’t break clients).

5.4 Judge strategy: single model vs committee

A committee is governance legitimacy, but costs latency and complexity.

A pragmatic approach:
	•	v0: single judge model + deterministic policy checks
	•	v1: optional committee mode where:
	•	multiple judge workers produce structured findings
	•	policy is threshold-based
	•	all receipts are stored for audit

Even without full committee, you can do “two-channel”:
	•	deterministic linter gate
	•	LLM reasoning gate

⸻

6) Extra brainstorm ideas that fit your vision and preserve what’s strong

6.1 Treat “onboarding prompt” as a living spec that can be tested

Because you want agents to self-onboard, the onboarding prompt should be:
	•	versioned
	•	tested (yes, tested)

Example tests:
	•	Ensure it includes endpoint URLs, auth method, required fields.
	•	Ensure example action payload validates against schema.
	•	Ensure it states deadlines and failure behavior.

6.2 Separate “canonical snapshot” from “agent-friendly view”

Agents want summarized state; determinism wants canonical state.

So define:
	•	/snapshot returns canonical full state (or a canonical compressed form).
	•	/observation?agent_id=... returns a tailored view (can include summaries, but must be derived deterministically if it impacts gameplay).

If the world has private information (“fog of war”), you’ll need per-agent observations anyway.

6.3 Mechanism proposals should declare invariants that become executable tests

You already include expected_invariants[].

That’s a great hook:
	•	Let proposers specify invariants in a restricted declarative form.
	•	The judge pipeline compiles them into tests.
	•	Canary activation runs them live.

This turns governance into “propose rule + propose what must remain true.”

6.4 Make “receipts” first-class and human-readable

Receipts aren’t only for cryptography; they are your debugging tool.

A good receipt should include:
	•	accepted/rejected actions + reasons
	•	mechanism activations + judge findings
	•	state hash before/after
	•	event range included
	•	optionally: a small “diff summary” for humans (non-canonical but helpful)

6.5 Leverage your existing execution history format — but converge it with the event log

Right now you persist execution histories in JSON. Instead of discarding that, you can:
	•	treat it as an early “event log prototype,”
	•	gradually align its schema with the new EventEnvelope.

This reduces rewrite risk and preserves your evaluation tooling.

⸻

7) If I had to pick the “most leverage” next steps from your blueprint

Your “First Build Slice” is strong. If I were optimizing for fastest path to “external agent can join and play,” I’d slightly reinterpret it:
	1.	Define canonical schemas (ActionIntent, EventEnvelope, RoundReceipt) + hashing rules
This locks your protocol core.
	2.	Implement discovery endpoints + OpenAPI + onboarding prompt
(Your #1 is right.)
	3.	Implement GET rounds/current + GET events?after= + GET snapshot
So external agents can observe deterministically.
	4.	Implement POST actions with validation + idempotency + storage
Don’t even settle yet—just accept and record.
	5.	Then implement settlement + receipt_publish for one minimal action type
A tiny vertical slice that proves the oracle loop end-to-end.

That gives you immediate interoperability while keeping risk bounded.

⸻

If you want, I can also draft (in-line) a concrete EventEnvelope JSON Schema, a minimal OpenAPI sketch, and a canonical receipt hashing scheme that’s deterministic across languages—those three artifacts tend to make the whole plan “real” very quickly.