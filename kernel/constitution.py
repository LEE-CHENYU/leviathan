"""Three-layer constitution: kernel (immutable), governance (amendable), world rules (open)."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


_DEFAULT_YAML = Path(__file__).parent / "constitution_default.yaml"


@dataclass
class Constitution:
    kernel_clauses: Dict[str, str] = field(default_factory=dict)
    governance_clauses: Dict[str, str] = field(default_factory=dict)
    world_rules: Dict[str, str] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def default(cls) -> "Constitution":
        return load_constitution(str(_DEFAULT_YAML))

    def compute_hash(self) -> str:
        canonical = json.dumps(
            {
                "kernel_clauses": self.kernel_clauses,
                "governance_clauses": self.governance_clauses,
                "world_rules": self.world_rules,
                "version": self.version,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def amend_governance(self, key: str, value: str) -> None:
        self.governance_clauses[key] = value
        self.version += 1

    def amend_world_rules(self, key: str, value: str) -> None:
        self.world_rules[key] = value
        self.version += 1

    def amend_kernel(self, key: str, value: str) -> None:
        raise ValueError("Kernel clauses are immutable and cannot be amended")


def load_constitution(path: str) -> Constitution:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Constitution(
        kernel_clauses=data.get("kernel_clauses", {}),
        governance_clauses=data.get("governance_clauses", {}),
        world_rules=data.get("world_rules", {}),
    )
