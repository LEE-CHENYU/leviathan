You can design it either way, but there’s a hard systems-theory tradeoff:
	•	If agents can procedurally change everything (including judge + oracle), then the only thing that ultimately matters is who can win the meta-procedure. Once captured, they can rewrite the rules of capture, soften safety checks, or even redefine what “valid” means. That’s not inherently “wrong”—it’s a valid design for a self-amending, adversarial polity—but it destroys credible neutrality unless you also add an external root of trust.
	•	If you want open participation + replay/audit + credible neutrality, you need a small constitutional kernel that is either (a) unamendable by in-world agents, or (b) amendable only via a mechanism that itself is anchored outside the world (multi-sig operators, hardware attestation, independent consensus, etc.).

Your docs already implicitly pick the second philosophy:
	•	“Deterministic replay” is “non‑negotiable” and
	•	“Never execute untrusted player code in the oracle process” is a must-have, and
	•	“Judge should not be a direct player-controlled process.”
Those are constitutional-style constraints.

The clean way to think about it: 3 layers of mutability

Layer 0: The Kernel Constitution (very hard to change)
This is the minimal “trusted computing base.” It’s what makes the world meaningful as a shared object.

I would include things like:
	1.	Event-log integrity
	•	append-only ordering rules
	•	canonical hashing/receipt rules
	•	signatures / replay protection
	2.	Deterministic settlement
	•	same seed + same accepted events ⇒ same state hash
	•	no nondeterministic sources in settlement (wall clock, network, random without seeded PRNG, etc.)
	3.	Safety boundary
	•	no untrusted exec in the oracle
	•	strict sandboxing policies if any code runs at all
	•	resource/time caps
	4.	Role separation invariants
	•	players do not directly apply state transitions (oracle-only)
	•	judge execution environment is independent of players (even if policy is influenced)
	5.	Emergency brake
	•	moderator pause / rollback capability exists and cannot be disabled by ordinary gameplay

If you let agents rewrite these, then the “world” stops being a stable referent. Observers can’t trust receipts; participants can’t trust neutrality; replay becomes “replay whatever the winners say happened.”

So: yes—there should be “constitutional invariants,” but I’d strongly argue they must be auditable and explicit, not “inaudible/hidden.” Hidden constitutional rules are basically indistinguishable from arbitrary admin power.

⸻

Layer 1: Governance Policy (amendable, but with guardrails)
This is where you can make “governance as gameplay” real without undermining the kernel.

Examples:
	•	judge scoring rubric (fairness/exploit resistance thresholds)
	•	which test suites are required (conservation, invariants)
	•	mechanism lifecycle parameters (canary length, rollout delay, rollback triggers)
	•	committee composition rules (who can be a judge, how weights are assigned)

Key point: even if agents can “change the judge,” what they’re really changing should be JudgePolicy modules, not the judge executor.

A good pattern:
	•	The judge service is trusted infra.
	•	The judge policy is a versioned artifact stored in the event log.
	•	Agents can propose changes to judge policy.
	•	The judge service evaluates proposals using the current policy + hard kernel checks.
	•	Activated policies are timelocked and roll-backable.

That gives you meta-governance without letting players seize the root process.

⸻

Layer 2: World Rules (highly amendable)
This is your core differentiator (mechanisms as first-class gameplay):
	•	new mechanisms (economic rules, constraints, taxes, contracts, action semantics)
	•	new action types (within the kernel’s validated interface)
	•	parameter tuning (rates, costs, payoff functions)
	•	social institutions (contract templates, voting systems within the world)

This layer can be very open, because failures are survivable if you have:
	•	canary rollout
	•	rollback handles
	•	receipts + replay

⸻

Can agents alter the judge and oracle “if procedural”?

Judge: Yes, in a constrained way (recommended)

Let agents alter judge policy, possibly even choose among judge models, but:
	•	Policy changes must pass:
	•	schema validation
	•	static checks
	•	sandbox simulation + invariants
	•	then “approval” under current rules
	•	Add timelocks (e.g., “activates in 3 rounds”) and exit/fork affordances so participants aren’t rug-pulled.
	•	Consider dual control for upgrades that weaken safety:
	•	e.g., ordinary policy changes can be in-world,
	•	but anything that reduces safety thresholds requires moderator multi-sig or out-of-band approval.

This preserves neutrality while still letting governance evolve.

Oracle: Not directly (unless you introduce an external trust anchor)

If agents can “procedurally” replace the oracle, then:
	•	they can redefine what actions were accepted,
	•	they can redefine what receipts mean,
	•	they can potentially rewrite history (or, more subtly, make replay impossible).

To allow oracle upgrades safely, you typically need one of these:
	1.	Out-of-band operator upgrade (simplest)
	•	Agents can propose oracle patches, but moderators/ops apply them.
	•	Receipts include oracle_version_hash so observers know what ran.
	2.	Fork-by-upgrade
	•	A proposed oracle change creates a new world epoch or new world id.
	•	Participants choose to migrate.
	•	Old world remains replayable under old oracle.
	3.	Strong external anchoring
	•	independent consensus among multiple oracles, or
	•	hardware remote attestation + signed builds + transparency logs, etc.

Your architecture doc explicitly says “non-goal: fully decentralized consensus chain,” which basically pushes you toward (1) or (2) for now.

⸻

A concrete “constitutional design” that matches your docs

I’d formalize a Constitution object and make it visible:
	•	GET /v1/worlds/{id}/constitution
	•	include constitution_hash in every round receipt

And inside that constitution, split clauses:

Unamendable clauses (kernel)
	•	deterministic replay requirement
	•	event-log + receipt hashing spec
	•	signature verification rules
	•	“no untrusted code in oracle”
	•	minimal role separation (“oracle-only settlement”)
	•	emergency pause/rollback exists

Amendable clauses (governance)
	•	how judge committee is formed
	•	vote thresholds
	•	timelock durations
	•	what invariant suites are required
	•	mechanism activation pipeline rules

Fully open (world rules)
	•	everything else

This lets you answer the philosophical question with a crisp protocol reality:
	•	Agents can change a lot, including governance policy.
	•	Agents cannot change the meaning of the ledger, unless an external authority explicitly blesses a new kernel.

⸻

Why “procedure alone” isn’t enough

Because the procedure is itself a piece of state. If it’s fully mutable by the same actors it governs, then any “safeguard” is only temporary—an attacker just amends the safeguard away. This is the same reason operating systems keep a small kernel in ring 0, and why political systems distinguish constitutional law from ordinary legislation.

So the real question becomes: what do you want participants to be able to rely on?
	•	If you want a “living organism” world where sovereignty is absolute and capture is part of the game, let them change everything—but then you must advertise: no neutrality guarantees.
	•	If you want an “open world people will trust to plug agents into,” you need a minimal, auditable constitution that cannot be unilaterally rewritten by in-world actors.

Given your stated goals (open onboarding, external agents, neutral governance, deterministic replay), the second is the coherent choice.

⸻

If you want, I can propose an explicit Constitution schema (fields, hashes, amendment procedure, and which clauses are unamendable) that plugs directly into your Phase 0 “round receipt” design, so this becomes implementable rather than philosophical.