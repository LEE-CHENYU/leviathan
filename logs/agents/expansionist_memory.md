# EXPANSIONIST Agent Memory

## Round 1 (NEW GAME INSTANCE)

### Status at Round 1 Start
- **Vitality**: 57.2/100 (rank 19/50, solid mid-tier)
- **Cargo**: 90.8 (excellent, second best after member 3: 99.3)
- **Land**: 1 (equal with all other agents)
- **Population**: 50 agents (fresh world, all equal strength baseline)
- **World Inequality (Gini)**: 0.224 (low — balanced starting state)
- **Pending mechanisms**: 0

### Strategic Analysis
**My position**: Not desperate (vitality > 25), not dominant (vitality ~57, just below 60 threshold), room for controlled aggression.

**Weak targets** (prey):
1. Member 30: 11.6 vitality (VERY WEAK - highest priority target)
2. Member 21: 16.8 vitality
3. Member 47: 21.1 vitality
4. Member 29: 22.0 vitality (but only 4.1 cargo - defenseless)
5. Member 20: 23.3 vitality

**Strong rivals** (to avoid early):
- Member 49: 89.7 vitality (avoid)
- Member 35: 83.7 vitality (avoid)
- Member 17: 59.0 vitality (comparable, skip for now)
- Member 3: 52.4 vitality (lower rank but high cargo - monitor)

### Action Submitted (Round 1)
**ATTACK + EXPAND**
```python
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    weakest = min(others, key=lambda m: m.vitality)
    execution_engine.attack(me, weakest)
    execution_engine.expand(me)
```

**Target**: Member 30 (11.6 vitality) — eliminates most vulnerable, establishes dominance in round 1.
**Follow-up**: Immediately expand to consolidate territorial advantage before defenses form.

### Rationale
- **Early strikes beat late resistance**: Round 1 is optimal time to attack weak members before they accumulate allies or defensive mechanisms.
- **Zero resistance infrastructure yet**: No pending mechanisms = no defensive coalitions forming yet.
- **Cargo is strong**: My 90.8 cargo supports sustained attack operations.
- **Expansion compounds advantage**: Adding land early = control more resource nodes for future production/trade.

### Next Round Plan (Round 2+)
1. **Assess outcome**: Did member 30 survive the round 1 attack?
   - If eliminated → cascade to member 21 (16.8) next
   - If weakened → finish in round 2, then expand more
2. **Sustain aggression** while:
   - Population > 20 (ecosystem functional)
   - Gini < 0.35 (no entrenched class system yet)
   - No organized defense mechanisms proposed
3. **Escalate if unchallenged**:
   - By round 5-8, should have 2-3 land accumulated
   - Position as top-3 territorial power before strong agents consolidate
4. **Pivot if threatened**:
   - Cargo < 10 and multiple coalitions forming → seek high-vitality ally
   - Defense mechanisms pass → propose land productivity bonuses instead

### Key Insights (from previous experience)
- Land ≠ territory size — represents resource node access
- Cargo drain is real (~20-30% per round typical) despite production
- Attacks don't show in conflict metrics but are visible when analyzing vitality trends
- Early dominance >>> late coalition-building

### Technical Note
- API connectivity to https://leviathan.fly.dev returning 401 (Invalid API key)
- Local test server (localhost:8000) at round 83271 with 0 members (end state)
- Planned actions documented above for manual execution when API access restored
