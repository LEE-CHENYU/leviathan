"""Minimal phase-graph utilities for the kernel runtime."""

from __future__ import annotations

from dataclasses import dataclass

from leviathan_core.protocols import PhaseGraphSpec, PhaseSpec


@dataclass
class PhaseGraph:
    """Topologically ordered phase graph."""

    spec: PhaseGraphSpec

    def __post_init__(self) -> None:
        self._phase_map = {phase.phase_id: phase for phase in self.spec.phases}
        if len(self._phase_map) != len(self.spec.phases):
            raise ValueError("Duplicate phase ids are not allowed")
        self._layers = self._build_layers()

    @property
    def phases(self) -> dict[str, PhaseSpec]:
        return dict(self._phase_map)

    @property
    def layers(self) -> list[list[PhaseSpec]]:
        return [list(layer) for layer in self._layers]

    def _build_layers(self) -> list[list[PhaseSpec]]:
        if not self.spec.phases:
            return []

        for phase in self.spec.phases:
            missing = [dep for dep in phase.depends_on if dep not in self._phase_map]
            if missing:
                raise ValueError(
                    f"Phase '{phase.phase_id}' depends on unknown phases: {missing}"
                )

        remaining = {
            phase.phase_id: set(phase.depends_on)
            for phase in self.spec.phases
        }
        layers: list[list[PhaseSpec]] = []

        while remaining:
            ready_ids = sorted(
                phase_id for phase_id, deps in remaining.items() if not deps
            )
            if not ready_ids:
                cycle_nodes = ", ".join(sorted(remaining))
                raise ValueError(f"Phase graph contains a cycle: {cycle_nodes}")

            layer = [self._phase_map[phase_id] for phase_id in ready_ids]
            layers.append(layer)

            for phase_id in ready_ids:
                del remaining[phase_id]
            for deps in remaining.values():
                deps.difference_update(ready_ids)

        return layers
