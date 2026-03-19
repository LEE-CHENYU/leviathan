"""Simple corporate-strategy flow built on top of ``leviathan_core``.

This is intentionally smaller than the full ``corporate_sim`` package in the
securities-selection repo. The purpose is to prove the kernel boundary with a
domain that has:

- multiple competing actors
- hidden action submission
- deterministic budget validation
- state reduction
- post-round evaluation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any

from leviathan_core import (
    ActionParser,
    AgentRole,
    Evaluator,
    JudgeAdapter,
    ObservationBuilder,
    PhaseGraphSpec,
    PhaseSpec,
    PluginCapabilities,
    Reducer,
)


def _as_nonnegative_number(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class CompanySnapshot:
    """Minimal company state for the demo plugin."""

    ticker: str
    revenue: float
    cash: float
    free_cash_flow: float
    rd_expense: float
    capex: float
    buyback_amount: float
    market_cap: float

    def available_budget(self) -> float:
        return max(0.0, self.free_cash_flow + self.cash * 0.5)

    def to_observation(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "revenue": self.revenue,
            "cash": self.cash,
            "free_cash_flow": self.free_cash_flow,
            "rd_expense": self.rd_expense,
            "capex": self.capex,
            "buyback_amount": self.buyback_amount,
            "market_cap": self.market_cap,
            "available_budget": self.available_budget(),
        }


class CorporateObservationBuilder(ObservationBuilder):
    """Shows each company its own full state and peer summaries."""

    def build(self, state: dict[str, Any], actor_id: str, round_ctx) -> dict[str, Any]:
        companies: dict[str, CompanySnapshot] = state["companies"]
        company = companies[actor_id]
        peers = {
            ticker: {
                "ticker": ticker,
                "revenue": peer.revenue,
                "market_cap": peer.market_cap,
                "free_cash_flow": peer.free_cash_flow,
            }
            for ticker, peer in companies.items()
            if ticker != actor_id
        }
        return {
            "round_index": round_ctx.round_index,
            "company": company.to_observation(),
            "competitors": peers,
        }


class CorporateActionParser(ActionParser):
    """Parses annual capital-allocation decisions."""

    _SCHEMA = """Return exactly one JSON object with this schema:
{
  "capital_allocation": {
    "rd_budget": <number>,
    "capex_budget": <number>,
    "buyback_budget": <number>
  },
  "strategic_moves": ["<short free-text move>", "..."],
  "rationale": "<short explanation>"
}"""

    def prompt_schema(self, actor_id: str, phase_id: str) -> str:
        return (
            f"You are management for {actor_id} in phase {phase_id}. "
            "Allocate capital for one strategic round. "
            f"{self._SCHEMA}"
        )

    def parse(self, raw_text: str, actor_id: str, phase_id: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned, count=1)
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first == -1 or last == -1 or last <= first:
                raise
            parsed = json.loads(cleaned[first : last + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Action must parse to a dict")
        return parsed

    def fallback(self, actor_id: str, phase_id: str) -> dict[str, Any]:
        return {
            "capital_allocation": {
                "rd_budget": 0.0,
                "capex_budget": 0.0,
                "buyback_budget": 0.0,
            },
            "strategic_moves": [],
            "rationale": "fallback",
        }


class CapitalAllocationJudge(JudgeAdapter):
    """Deterministic finance guardrails for the demo plugin."""

    def validate(self, state: dict[str, Any], actions: dict[str, dict[str, Any]], round_ctx):
        companies: dict[str, CompanySnapshot] = state["companies"]
        adjusted: dict[str, dict[str, Any]] = {}
        report: dict[str, Any] = {}

        for ticker, action in actions.items():
            company = companies[ticker]
            alloc = action.get("capital_allocation", {})

            rd = _as_nonnegative_number(alloc.get("rd_budget", company.rd_expense))
            capex = _as_nonnegative_number(alloc.get("capex_budget", company.capex))
            buyback = _as_nonnegative_number(
                alloc.get("buyback_budget", company.buyback_amount)
            )

            rd_floor = company.rd_expense * 0.8
            available = company.available_budget()
            violations: list[dict[str, Any]] = []

            if rd < rd_floor:
                violations.append(
                    {
                        "constraint": "rd_floor",
                        "original": rd,
                        "adjusted": rd_floor,
                    }
                )
                rd = rd_floor

            total = rd + capex + buyback
            if total > available:
                original_total = total
                excess = total - available

                buyback_cut = min(buyback, excess)
                buyback -= buyback_cut
                excess -= buyback_cut

                if excess > 0:
                    capex_cut = min(capex, excess)
                    capex -= capex_cut
                    excess -= capex_cut

                if excess > 0:
                    rd_cut = min(max(0.0, rd - rd_floor), excess)
                    rd -= rd_cut
                    excess -= rd_cut

                violations.append(
                    {
                        "constraint": "budget",
                        "original": original_total,
                        "adjusted": rd + capex + buyback,
                        "available": available,
                    }
                )

            adjusted[ticker] = {
                **action,
                "capital_allocation": {
                    "rd_budget": round(rd, 2),
                    "capex_budget": round(capex, 2),
                    "buyback_budget": round(buyback, 2),
                },
            }
            report[ticker] = {
                "available_budget": round(available, 2),
                "validated_spend": round(rd + capex + buyback, 2),
                "violations": violations,
            }

        return adjusted, report


class CorporateStateReducer(Reducer):
    """Applies validated capital allocation to each company state."""

    def apply(self, state: dict[str, Any], actions: dict[str, dict[str, Any]], round_ctx):
        companies: dict[str, CompanySnapshot] = state["companies"]
        new_companies: dict[str, CompanySnapshot] = {}

        for ticker, company in companies.items():
            action = actions.get(ticker, {})
            alloc = action.get("capital_allocation", {})

            rd_budget = _as_nonnegative_number(alloc.get("rd_budget", company.rd_expense))
            capex_budget = _as_nonnegative_number(alloc.get("capex_budget", company.capex))
            buyback_budget = _as_nonnegative_number(
                alloc.get("buyback_budget", company.buyback_amount)
            )

            rd_ratio = rd_budget / max(company.rd_expense, 1.0)
            capex_ratio = capex_budget / max(company.capex, 1.0)
            growth = 0.02 + (rd_ratio - 1.0) * 0.04 + (capex_ratio - 1.0) * 0.03
            growth = max(-0.05, min(0.18, growth))

            new_revenue = company.revenue * (1.0 + growth)
            new_fcf = max(0.0, company.free_cash_flow * (0.95 + growth))
            new_cash = max(
                0.0,
                company.cash + company.free_cash_flow - rd_budget - capex_budget - buyback_budget,
            )
            valuation_shift = max(-0.2, min(0.25, growth + (0.02 if new_cash > 0 else -0.08)))
            new_market_cap = max(0.0, company.market_cap * (1.0 + valuation_shift))

            new_companies[ticker] = replace(
                company,
                revenue=round(new_revenue, 2),
                cash=round(new_cash, 2),
                free_cash_flow=round(new_fcf, 2),
                rd_expense=round(rd_budget, 2),
                capex=round(capex_budget, 2),
                buyback_amount=round(buyback_budget, 2),
                market_cap=round(new_market_cap, 2),
            )

        return {
            "companies": new_companies,
            "round_history": state.get("round_history", []) + [actions],
        }


class CorporateScoreEvaluator(Evaluator):
    """Produces a simple per-company management score."""

    async def evaluate(self, state: dict[str, Any], actions: dict[str, dict[str, Any]], round_ctx):
        scores: dict[str, dict[str, Any]] = {}
        companies: dict[str, CompanySnapshot] = state["companies"]

        for ticker, company in companies.items():
            alloc = actions.get(ticker, {}).get("capital_allocation", {})
            buyback = _as_nonnegative_number(alloc.get("buyback_budget", company.buyback_amount))
            rd = _as_nonnegative_number(alloc.get("rd_budget", company.rd_expense))

            score = 7.0
            if company.cash <= 0:
                score -= 2.0
            if buyback > company.available_budget() * 0.4:
                score -= 1.0
            if rd >= company.rd_expense:
                score += 0.5
            score = max(0.0, min(10.0, score))

            scores[ticker] = {
                "weighted_score": round(score, 2),
                "ending_cash": company.cash,
                "ending_revenue": company.revenue,
            }

        return scores


class SimpleCorporateStrategyPlugin:
    """Corporate-strategy plugin that exercises the kernel interfaces."""

    plugin_id = "corporate_strategy_demo"
    capabilities = PluginCapabilities(deterministic_judge=True, scoring_panel=True)

    def __init__(self) -> None:
        self._observation_builder = CorporateObservationBuilder()
        self._action_parser = CorporateActionParser()
        self._judge = CapitalAllocationJudge()
        self._reducer = CorporateStateReducer()
        self._evaluator = CorporateScoreEvaluator()

    def build_initial_state(self, config: dict[str, Any]) -> dict[str, Any]:
        companies_config = config.get("companies", [])
        companies = {
            company["ticker"]: CompanySnapshot(**company)
            for company in companies_config
        }
        return {
            "companies": companies,
            "round_history": [],
        }

    def build_phase_graph(self, config: dict[str, Any]) -> PhaseGraphSpec:
        return PhaseGraphSpec(
            phases=[
                PhaseSpec("strategy", "action", hidden_actions=True),
                PhaseSpec("judge", "judge", depends_on=["strategy"]),
                PhaseSpec("market", "reduce", depends_on=["judge"]),
                PhaseSpec("score", "evaluate", depends_on=["market"]),
            ]
        )

    def get_roles(self, state: dict[str, Any]) -> list[AgentRole]:
        return [
            AgentRole(role_id=ticker, label=f"{ticker} Management", kind="management")
            for ticker in sorted(state["companies"])
        ]

    def get_observation_builder(self, phase_id: str) -> ObservationBuilder | None:
        return self._observation_builder if phase_id == "strategy" else None

    def get_action_parser(self, phase_id: str) -> ActionParser | None:
        return self._action_parser if phase_id == "strategy" else None

    def get_judge(self, phase_id: str) -> JudgeAdapter | None:
        return self._judge if phase_id == "judge" else None

    def get_reducer(self, phase_id: str) -> Reducer | None:
        return self._reducer if phase_id == "market" else None

    def get_evaluator(self, phase_id: str) -> Evaluator | None:
        return self._evaluator if phase_id == "score" else None

    def get_stop_condition(self):
        return None

    def get_finalizer(self):
        return None

    @staticmethod
    def default_config() -> dict[str, Any]:
        """Convenience config for tests and demos."""

        return {
            "rounds": 1,
            "companies": [
                {
                    "ticker": "AAA",
                    "revenue": 1000.0,
                    "cash": 300.0,
                    "free_cash_flow": 120.0,
                    "rd_expense": 80.0,
                    "capex": 50.0,
                    "buyback_amount": 20.0,
                    "market_cap": 2500.0,
                },
                {
                    "ticker": "BBB",
                    "revenue": 850.0,
                    "cash": 180.0,
                    "free_cash_flow": 60.0,
                    "rd_expense": 70.0,
                    "capex": 45.0,
                    "buyback_amount": 10.0,
                    "market_cap": 2100.0,
                },
            ],
        }

    @staticmethod
    def state_to_dict(state: dict[str, Any]) -> dict[str, Any]:
        """Convert state to plain dicts for inspection."""

        return {
            "companies": {
                ticker: asdict(company)
                for ticker, company in state["companies"].items()
            },
            "round_history": state.get("round_history", []),
        }
