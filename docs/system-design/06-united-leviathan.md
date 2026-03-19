Yes—multiple Leviathans on different servers is absolutely feasible.
What gets “hard” (or not) depends on what you mean by aggregate into a larger Leviathan.

In distributed sims like yours (round-based, deterministic, event-sourced), there are three qualitatively different “aggregation” models. Only one of them is truly “one big world.”

⸻

1) Three ways to “wire multiple Leviathans together”

A) Directory aggregation (many worlds; one universe catalog)

Meaning: each server runs its own Leviathan world(s). Clients join one. A “universe” service aggregates discovery, identity, and analytics.
	•	Feasibility: very high
	•	Complexity: low → medium
	•	What you get: horizontal scaling by adding worlds, operational isolation, easy experimentation
	•	What you don’t get: a single shared world state—this is a federation of worlds, not one big world.

This fits your existing plan well because your blueprint already has GET /v1/worlds, discovery manifests, capabilities, and multi-tenant controls. You basically add a World Directory / Router on top.

Good for: “we want many parallel Leviathans, and want them to feel like one ecosystem.”

⸻

B) Federated universe (many sovereign worlds with cross-world travel/messages/assets)

Meaning: each server is authoritative for its own world, but worlds can exchange “bridge events” (travel, messaging, asset transfer). Think “multiple realms with portals.”
	•	Feasibility: high, but the devil is in bridge semantics
	•	Complexity: medium → high (depends on whether you need atomic transfers)
	•	Core design move: cross-world effects are asynchronous, event-based, and applied at a deterministic phase boundary (e.g., snapshot_open).

This is the sweet spot if you want “larger Leviathan” as a network of worlds rather than one globally consistent database.

Good for: “agents can migrate, trade, form alliances across servers,” while keeping each world deterministic and independently replayable.

⸻

C) Single world sharded across servers (one global Leviathan state)

Meaning: there is one world, but its state is partitioned across servers (shards). Clients connect anywhere; actions can affect entities across shards in the same round.
	•	Feasibility: yes in principle
	•	Complexity: very high (this is basically “build a distributed database + distributed deterministic simulation”)
	•	Why it’s hard: you need global ordering + cross-shard transaction semantics while preserving your non‑negotiables:
	•	deterministic replay
	•	canonical event ordering
	•	clean trust boundary (no untrusted code in oracle)
	•	phase deadlines

If “aggregate into a larger Leviathan” means “one single coherent world where any agent can interact with any other agent seamlessly,” this is the model. But it’s the most expensive one by far.

Good for: when you’re truly bottlenecked by a single oracle and need one massive world—not as your first distributed milestone.

⸻

2) There’s a 4th thing people confuse with “aggregation”: HA replication

D) One world, replicated for high availability (not bigger, just more reliable)

Meaning: multiple servers run copies, but only one leader orders writes (or you use a consensus log). Followers serve reads and can take over.
	•	Feasibility: high
	•	Complexity: medium
	•	Benefit: “clients can join via different servers,” but it’s still one world.

This is often what you want early if your concern is “multi-server” operationally, not “bigger world.”

⸻

3) Complexity ladder (so you can decide without ambiguity)

If I rank these by implementation complexity:
	1.	Directory aggregation (A)
	2.	HA replication (D)
	3.	Federated universe (B)
	4.	Sharded single world (C)

The big jump is (B)→(C).
(C) is where you start paying the “distributed transactions + global ordering” tax.

⸻

4) What makes wiring worlds together hard in your architecture

Your docs emphasize:
	•	deterministic phase execution + canonical event ordering
	•	append-only event log + receipts
	•	external agents + strict trust boundary
	•	“non-goal: fully decentralized consensus chain”

Those constraints strongly push you toward designs where:
	•	each world has a single authoritative oracle for settlement, and
	•	cross-server composition happens via events/receipts, not shared mutable state.

The moment you want “one big world” across servers, you are effectively building a global ordering system for all actions/events. That’s either:
	•	a leader log (central sequencer),
	•	or a consensus group (Raft-like),
	•	or a full decentralization scheme (which you explicitly don’t want right now).

So: feasible, but expensive.

⸻

5) Practical recommendation: how to get “larger Leviathan” without the hardest part

If your goal is “it feels like one big Leviathan ecosystem,” you can get 80% of the experience with A + B:
	•	A Universe Directory to discover worlds
	•	Optional Cross-world bridges for migration / messaging / asset transfer
	•	Keep each world deterministic and replayable

This gives you:
	•	scale-out by adding servers/worlds
	•	natural isolation boundaries (abuse in one world doesn’t kill all)
	•	governance experimentation (different constitutions/policies per world)
	•	a clean path to “bigger” without sharding complexity

⸻

6) Preparation work you can do now (even before you fully implement federation/sharding)

These are “no-regret” steps: they help the single-server distributed version and any future multi-server wiring.

Prep 1: Make receipts + events fully canonical and portable

You already plan RoundReceipt with state hashes and action hashes. Extend that mindset slightly:
	•	Canonical event envelope must include:
	•	world_id
	•	monotonic event_seq
	•	round_id
	•	phase
	•	type
	•	payload_hash
	•	prev_event_hash (optional but powerful)
	•	A deterministic, language-independent hashing/encoding rule (canonical JSON or stable binary encoding)

This is the foundation of:
	•	multi-server replication (D)
	•	federation proofs (B)
	•	sharded ordering (C)

Prep 2: Give each world an oracle signing identity

Even if everything is run by you initially, add:
	•	world_public_key
	•	receipts signed by the world oracle key

Then cross-world bridges can do:
	•	“I accept event X from world A if it is signed and references a valid receipt.”

This is the difference between “random HTTP call” and “verifiable inter-world claim.”

Prep 3: Reserve protocol fields for inter-world composition

Even if unused at first, add fields now so you don’t break clients later:
	•	origin_world_id
	•	origin_receipt_hash
	•	bridge_channel_id
	•	bridge_seq
	•	notary_signature (optional)

Prep 4: Explicitly define where “external inputs” enter the round

For federation, you want deterministic ingestion:
	•	Add a phase boundary such as inbox_ingest (or treat it as part of snapshot_open)
	•	Rule: only inbound bridge events received before deadline D are included in round R

That keeps replay stable and makes network delays predictable (they just shift to next round).

Prep 5: Build a multi-world simulation harness locally

Before doing real federation, do the “boring but decisive” test harness:
	•	spin up 2 world servers locally (docker compose)
	•	run:
	•	independent rounds
	•	cross-world message injection
	•	verify both worlds replay deterministically
	•	include fault cases:
	•	duplicated bridge message (idempotency)
	•	delayed delivery
	•	partial failure mid-transfer

This turns a theoretical protocol into something you can validate.

Prep 6: Decide early what “bridge atomicity” you need

This is the main fork in complexity for federation (B):
	•	Non-atomic bridge (easy): “message arrives; world reacts.”
Great for travel requests, diplomacy, gossip, announcements.
	•	Atomic asset transfer (harder): prevent double-spend across worlds.
Needs either:
	•	a notary/bridge service (central coordinator), or
	•	a two-phase escrow protocol with timeouts + proofs, plus a trust model.

If you keep v0 bridges message-only, federation becomes much simpler.

⸻

7) If you do want “one big Leviathan” (single world) later: a realistic architecture shape

If you eventually want model (C), the least painful way is usually:
	1.	Central Sequencer (Global Event Orderer)
One service assigns global order to all action intents across shards.
	2.	Shard-local settlement with deterministic cross-shard references
Most actions touch one shard; cross-shard actions become:
	•	(a) scheduled for next round, or
	•	(b) resolved via deterministic two-phase protocol
	3.	Strict limitations on cross-shard interactions early
e.g., you can’t directly “trade” across shards in the same round; you must use an escrow mechanism that completes next round.

This is still complex, but it avoids full distributed transactions on day one.

⸻

Bottom line
	•	Feasible? Yes.
	•	Is it complex?
	•	If “aggregate” = directory + federation → manageable (buildable with your current blueprint trajectory).
	•	If “aggregate” = one coherent sharded world → very complex, and you should plan it as a later stage.
	•	Best preparation work now: canonical receipts/events, world signing identity, deterministic inbox phase, reserved bridge fields, and a two-world harness with idempotency + replay tests.

If you tell me which “aggregation” you mean (directory vs federation vs sharded single world vs HA replication), I can sketch a concrete protocol shape and the minimal set of new endpoints/events you’d need—but even without that, the prep list above is the right groundwork to start immediately.